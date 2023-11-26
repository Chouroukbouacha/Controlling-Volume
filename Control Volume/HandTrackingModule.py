import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode = False, maxHands = 1, modelComplexity = 1, detectionConfidence = 0.5, trackingConfidence = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionConfidence = detectionConfidence
        self.trackingConfidence = trackingConfidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionConfidence, self.trackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils
    
    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
 
        if self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                    self.mpDraw.draw_landmarks(img, hand_landmarks, self.mpHands.HAND_CONNECTIONS)
    
        return img
    
    def findPosition(self, img, handNumber = 0, draw = True):
        
        landmark_list = []
        
        if self.results.multi_hand_landmarks:
            hand_landmarks = self.results.multi_hand_landmarks[handNumber]
            
            # Process only the landmarks we need: thumb tip (ID 4) and index finger tip (ID 8)
            for id, landmark in enumerate(hand_landmarks.landmark):
                if id == 4 or id == 8:
                    h, w, c = img.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    landmark_list.append([id, cx, cy])
                    
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

            # Draw the line connecting these two points
            if draw and len(landmark_list) == 2:
                cv2.line(img, tuple(landmark_list[0][1:]), tuple(landmark_list[1][1:]), (255, 0, 0), 3)
            
        return landmark_list


def main():
    cap = cv2.VideoCapture(0)
    
    prev_time = 0
    curr_time = 0
    
    detector = handDetector()
    
    while True:
        success, img = cap.read()
        
        img = detector.findHands(img)
        
        # landmark_list = detector.findPosition(img)
        # if len(landmark_list) != 0:
        #     print(landmark_list[4])
    
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(img, f'fps: {int(fps)}', (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    main()