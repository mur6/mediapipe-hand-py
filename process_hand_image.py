
import cv2
import mediapipe as mp
import mediapipe.python.solutions as solutions

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = solutions.drawing_styles
mp_hands = mp.solutions.hands




def main(filename):
    with solutions.hands.Hands(
        static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
    ) as hands:

        # Read an image, flip it around y-axis for correct handedness output (see
        # above).
        image = cv2.flip(cv2.imread(filename), 1)
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print handedness and draw hand landmarks on the image.
        print("Handedness:", results.multi_handedness)
        if not results.multi_hand_landmarks:
            raise RuntimeError("Not found hand landmarks.")

        image_height, image_width, _ = image.shape
        annotated_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            #print("hand_landmarks:", hand_landmarks)
            print(
                f"Index finger tip coordinates: (",
                f"{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, "
                f"{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})",
            )
        print(len(hand_landmarks.landmark))

main("images/hand.jpeg")
