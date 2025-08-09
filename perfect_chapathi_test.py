import cv2
import numpy as np
import streamlit as st
from PIL import Image

def calculate_roundness(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0
    roundness = (4 * np.pi * area) / (perimeter * perimeter)
    return roundness * 100

def get_chapathi_mask(img):
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)

    # Normalize lighting on L-channel to reduce shadow effects
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_eq = clahe.apply(L)

    # Combine A and B channels into a single grayscale-like image
    ab_gray = cv2.addWeighted(A, 0.5, B, 0.5, 0)

    # Blur to smooth color edges
    ab_blur = cv2.GaussianBlur(ab_gray, (5, 5), 0)

    # Otsu threshold for chapathi segmentation
    _, mask = cv2.threshold(ab_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleanup to remove noise
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    return mask

def detect_toast_level(img, mask):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_eq = clahe.apply(L)

    brown_mask = cv2.inRange(L_eq, 90, 160)
    yellow_mask = cv2.inRange(B, 135, 200)

    combined = cv2.bitwise_and(brown_mask, yellow_mask)
    combined = cv2.bitwise_and(combined, combined, mask=mask)

    total = cv2.countNonZero(mask)
    browns = cv2.countNonZero(combined)
    if total == 0:
        return 0.0

    toast_percent = (browns / total) * 100
    return 30 + toast_percent

def detect_brown_spots(img, mask):
    # Convert to LAB
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)

    # Toasted spots are darker (low L) and more yellow-brown (high B)
    brown_mask = cv2.inRange(L, 0, 160)  # Lower lightness = darker
    yellow_mask = cv2.inRange(B, 135, 200)  # Yellowish-brown tones

    combined = cv2.bitwise_and(brown_mask, yellow_mask)
    combined = cv2.bitwise_and(combined, mask)

    # Clean noise
    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find brown spot contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    brown_spots = [cnt for cnt in contours if 20 < cv2.contourArea(cnt) < 2000]

    return brown_spots


def process_image(uploaded_image):
    img = np.array(uploaded_image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Shadow-proof chapathi segmentation
    mask = get_chapathi_mask(img)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img, 0, 0, 0

    biggest = max(contours, key=cv2.contourArea)
    roundness = calculate_roundness(biggest)

    toast_level = detect_toast_level(img, mask)
    brown_spots = detect_brown_spots(img, mask)

    output = img.copy()
    cv2.drawContours(output, [biggest], -1, (0, 255, 0), 5)

    for cnt in brown_spots:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return output, roundness, toast_level, len(brown_spots)

def get_rating_message(score):
    if score > 90:
        return "🥇 Perfect! Your chapathi deserves an award."
    elif score > 75:
        return "👏 Pretty round! Mom would be proud."
    elif score > 60:
        return "🙂 Almost there! Try a rolling pin with more love."
    elif score > 40:
        return "😬 Hmm... Artistic attempt?"
    else:
        return "🤡 Did you drop this on the floor and stamp it?"

def get_toast_message(toast_level):
    if toast_level < 20:
        return "🌞 Pale beauty — did you even light the stove?"
    elif toast_level < 40:
        return "🥱 Lightly kissed by the pan — elegant but shy."
    elif toast_level < 60:
        return "🍯 Golden perfection — Instagram-worthy!"
    elif toast_level < 80:
        return "🔥 A bit on the adventurous side — crunchy vibes!"
    else:
        return "💀 Charcoal edition — perfect for BBQ lovers."

# Streamlit UI
st.title("🫓 Chapathi Perfection Rater! ")
st.write("Upload a top-view image of your chapathi and prepare to be judged 😈.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Chapathi", width=300)

    result_img, roundness, toast_level, spot_count = process_image(image)
    with col2:
        st.image(result_img, caption=f"Detected Chapathi (Roundness: {roundness:.2f}%)", width=300)

    st.subheader("📊 Results:")
    st.markdown(f"**Roundness Score:** {roundness:.2f}%")
    st.markdown(f"**Toast Level %:** {toast_level:.2f}%")
    st.markdown(f"**Brown Spots Detected:** {spot_count}")

    st.subheader("Verdict:")
    st.markdown("##### 🔵 Roundness:")
    st.success(get_rating_message(roundness))
    st.markdown("##### 🍞 Toast:")
    st.success(get_toast_message(toast_level))
