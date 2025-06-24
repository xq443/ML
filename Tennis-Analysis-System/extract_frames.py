import os
import cv2

def extract_frames(video_path, fps_step=1, output_dir="frames"):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    timestamps = []
    frame_idx = 0
    saved_frame_idx = 0

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % fps_step == 0:
            timestamp = frame_idx / fps
            frame_filename = os.path.join(output_dir, f"frame_{saved_frame_idx:05d}.jpg")
            cv2.imwrite(frame_filename, frame)

            frames.append(frame)
            timestamps.append(timestamp)
            saved_frame_idx += 1

        frame_idx += 1

    cap.release()
    return frames, timestamps


# ----------------- Unit Test -----------------
if __name__ == "__main__":
    import numpy as np

    def test_extract_frames():
        print("Running unit test for extract_frames...")

        test_video_path = "output_videos/output.avi"
        output_dir = "extracted_frames"

        # Create dummy video if not exists
        if not os.path.exists(test_video_path):
            os.makedirs("output_videos", exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(test_video_path, fourcc, 10.0, (640, 480))
            for _ in range(30):
                black_frame = np.zeros((480, 640, 3), dtype='uint8')
                out.write(black_frame)
            out.release()

        # Call the updated function
        frames, timestamps = extract_frames(test_video_path, fps_step=10, output_dir=output_dir)

        # Basic assertions
        assert isinstance(frames, list)
        assert isinstance(timestamps, list)
        assert len(frames) == len(timestamps)
        assert all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))

        saved_files = [f for f in os.listdir(output_dir) if f.endswith('.jpg')]
        assert len(saved_files) == len(frames), "Mismatch between saved and extracted frames"

        print(f"✅ Extracted and saved {len(frames)} frames to '{output_dir}/'")
        print("Timestamps:", timestamps)
        print("Test passed!")

    test_extract_frames()
