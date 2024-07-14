import os
import torch
import numpy as np
import dlib
import librosa
from PIL import Image
from moviepy.editor import VideoFileClip
from torch.utils.data import DataLoader
from options.test_options import TestOptions
from data.base_dataset import get_transform, BaseDataset
from models import networks, DFD_model
from models.DFD_model import DFDModel
import argparse
import sys

detector = dlib.get_frontal_face_detector()

def extract_mel_spectrogram(audio_path, output_path):
    y, sr = librosa.load(audio_path, sr=16000)
    target_length = sr * 3  # 3 seconds
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), 'constant')
    else:
        y = y[: target_length]

    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=160, n_mels=512)
    mel_out = librosa.power_to_db(mel_spect, ref=np.max)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, mel_out)


def extract_frame_and_crop_face(video_path, output_path):
    cap = VideoFileClip(video_path)
    frames = cap.iter_frames()

    for frame in frames:
        img = Image.fromarray(frame)
        gray = img.convert('L')
        rects = detector(np.array(gray), 1)

        if len(rects) > 0:
            rect = max(rects, key=lambda r: r.width() * r.height())
            cropped_face = img.crop((rect.left(), rect.top(), rect.right(), rect.bottom()))
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cropped_face.save(output_path)
            return True

    print(f"No face detected in video {video_path}")
    return False


def process_single_video(video_path, output_root):
    audio_feat_dir = os.path.join(output_root, 'audio_feat')
    true_frames_dir = os.path.join(output_root, 'frame')
    os.makedirs(audio_feat_dir, exist_ok=True)
    os.makedirs(true_frames_dir, exist_ok=True)

    video_filename = os.path.basename(video_path)
    unique_filename = video_filename.replace('.mp4', '')

    audio_target_file = os.path.join(audio_feat_dir, unique_filename + '.npy')
    frame_target_file = os.path.join(true_frames_dir, unique_filename + '.jpg')
    # audio_test_file = os.path.join("/userhome/cs2/u3619712/FaceOff/evaluation/video_test", unique_filename + '.wav')
    audio_test_file = os.path.join("/userhome/cs2/u3619712/FaceOff/evaluation/video_eval", unique_filename + '.wav')

    video = VideoFileClip(video_path)
    audio = video.audio
    # audio_temp_file = unique_filename + '.wav'
    audio.write_audiofile(audio_test_file, logger=None)
    
    extract_mel_spectrogram(audio_test_file, audio_target_file)
    # os.remove(audio_temp_file)
    extract_frame_and_crop_face(video_path, frame_target_file)

    # print("Video processed successfully.")
    return frame_target_file, audio_target_file


def predict_real_or_fake(model, aud_real, img_real):
    model.set_test_input({'label': None, 'img': img_real, 'aud': aud_real})
    model.forward_test()
    sim_A_V, _ = model.val()

    return sim_A_V.item()

def main(video_path, output_root):
    frame_path, audio_path = process_single_video(video_path, output_root)


    # Instantiate DFDModel
    opt = TestOptions().parse()

    # opt.model = model_type
    # print(opt)
    # opt.no_flip = no_flip
    # opt.checkpoints_dir = checkpoints_dir
    # opt.name = name

    model = DFDModel(opt)
    model.setup(opt)

    # Prepare the input
    transform = get_transform(opt)

    image_input = Image.open(frame_path).convert('RGB')
    img_d = transform(image_input).unsqueeze(0).unsqueeze(0)  # (1, 3, H, W)
    # print(f"Image tensor shape: {img_d.shape}")

    audio = np.load(audio_path)
    audio_d = librosa.util.normalize(audio)
    audio_d = torch.tensor(audio_d).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 512, T)
    # print(f"Audio tensor shape : {audio_d.shape}")


    # Predict using DFDModel
    model.set_test_input({'label': None, 'img': img_d, 'aud': audio_d})

    model.img_fake = img_d
    model.aud_fake = audio_d

    model.forward_test()
    sim_A_V, _ = model.val()

    # print(f"Similarity score: {sim_A_V.item()}")

    # Binary decision based on threshold
    if sim_A_V.item() < threshold:
        print("True") # deepfake
    else:
        print("False")

    return sim_A_V.item()


if __name__ == '__main__':
    # print(sys.argv)

    # real
    # video_path = '/userhome/cs2/u3619603/FakeAVCeleb/RealVideo-RealAudio/Asian (South)/men/id00032/00028.mp4'
    video_path = sys.argv[2]
    # fake
    # video_path = '/userhome/cs2/u3619603/FakeAVCeleb/FakeVideo-RealAudio/Caucasian (American)/women/id00025/00025_id00098_wavtolip.mp4'
    
    output_root = 'Dataset/singleVideo'
    threshold = 815
    main(video_path, output_root)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="DFD detection script")
#     parser.add_argument('--model', type=str, default='DFD', help='chooses which model to use.')
#     parser.add_argument('--no_flip', action='store_true', help='Disable flipping')
#     parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='Checkpoints directory')
#     parser.add_argument('--name', type=str, default='VFD', help='Name of the experiment')

#     parser.add_argument('--threshold', type=float, default=815, help='Threshold for decision')
#     parser.add_argument('--video_path', type=str, required=True, help='Path to the input video')
#     parser.add_argument('--output_root', type=str, default='Dataset/singleVideo', help='Output root directory')

#     args = parser.parse_args()

#     print(args.model)

#     main(args.video_path, args.output_root, args.model, args.no_flip, args.checkpoints_dir, args.name, args.threshold)
