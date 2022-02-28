
import cv2
import mediapipe as mp
import mediapipe.python.solutions as solutions

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For static images:
IMAGE_FILES = []


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
            print("hand_landmarks:", hand_landmarks)
            print(
                f"Index finger tip coordinates: (",
                f"{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, "
                f"{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})",
            )
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )
        cv2.imwrite(
            "annotated_hand_image.png", cv2.flip(annotated_image, 1)
        )
        # # Draw hand world landmarks.
        # if not results.multi_hand_world_landmarks:
        #     raise 

        # for hand_world_landmarks in results.multi_hand_world_landmarks:
        #     mp_drawing.plot_landmarks(
        #         hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5
        #     )
main("images/hand.jpeg")