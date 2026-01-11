import cv2
from fer.fer import FER

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try changing VideoCapture(0) to (1) or (2).")

    detector = FER(mtcnn=False)

    analyze_every_n_frames = 10
    frame_i = 0

    last_top_emotion = "—"
    last_score = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)

        if frame_i % analyze_every_n_frames == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # returns list of detections, each has 'box' and 'emotions'
            results = detector.detect_emotions(rgb)

            if results:
                # pick the largest face
                best = max(results, key=lambda r: r["box"][2] * r["box"][3])
                (x, y, w, h) = best["box"]
                emotions = best["emotions"]

                # top emotion
                top_emotion, top_score = max(emotions.items(), key=lambda kv: kv[1])
                last_top_emotion = top_emotion
                last_score = float(top_score)

                # draw face box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # draw a small bar chart of emotions
                y0 = max(0, y - 10)
                x0 = x + w + 10
                bar_w = 120
                line_h = 18

                for idx, (emo, score) in enumerate(sorted(emotions.items(), key=lambda kv: kv[1], reverse=True)):
                    if idx >= 5:  # show top 5 only
                        break
                    text = f"{emo}: {score:.2f}"
                    yy = y0 + idx * line_h
                    cv2.putText(frame, text, (x0, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.rectangle(frame, (x0, yy + 4), (x0 + int(bar_w * score), yy + 12), (255, 255, 255), -1)

        # overlay top emotion always (even between analyzed frames)
        cv2.putText(
            frame,
            f"Emotion: {last_top_emotion} ({last_score:.2f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )

        cv2.imshow("FER Emotion Detector (press q to quit)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        frame_i += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
