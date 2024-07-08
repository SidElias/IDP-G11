import json
import os
from pytube import YouTube
from concurrent.futures import ThreadPoolExecutor, as_completed

# Function to read JSON file and extract YouTube links
def read_json_and_extract_links(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    print(f"Loaded data from {file_path}:")
    print(json.dumps(data, indent=2))  # Print the JSON data to inspect its structure
    
    youtube_links = []
    for item in data:
        if 'url' in item:
            youtube_links.append(item['url'])
        else:
            print(f"Warning: 'url' key not found in {item}")
    return youtube_links

# Function to download a single video
def download_video(link, output_directory):
    try:
        yt = YouTube(link)
        video = yt.streams.get_highest_resolution()
        # Construct the expected filename
        filename = video.default_filename
        file_path = os.path.join(output_directory, filename)
        if not os.path.exists(file_path):
            print(f"Downloading {link} to {output_directory}")
            video.download(output_directory)
        else:
            print(f"Skipping {link} (already downloaded)")
    except Exception as e:
        print(f"Error downloading {link}: {e}")

# Function to download videos from YouTube links using multithreading
def download_videos_multithreaded(youtube_links, output_directory, max_workers=4):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_video, link, output_directory) for link in youtube_links]
        for future in as_completed(futures):
            future.result()

# Directory containing JSON files
json_directory = r'E:\Semester 5\Integrated Design Project\Test\Dataset'
train_json_path = os.path.join(json_directory, 'MSASL_train.json')
validate_json_path = os.path.join(json_directory, 'MSASL_val.json')
test_json_path = os.path.join(json_directory, 'MSASL_test.json')

# Read JSON files and extract YouTube links
train_links = read_json_and_extract_links(train_json_path)
validate_links = read_json_and_extract_links(validate_json_path)
test_links = read_json_and_extract_links(test_json_path)

# Download videos to specific directories
download_videos_multithreaded(train_links, 'videos/train', max_workers=8)
download_videos_multithreaded(validate_links, 'videos/validate', max_workers=8)
download_videos_multithreaded(test_links, 'videos/test',max_workers=8)