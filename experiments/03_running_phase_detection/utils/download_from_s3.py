import boto3
import os
import argparse

def download_s3_folder(s3_uri, local_directory):
    """
    Download the JPG files from a folder in S3 to a local directory.

    Parameters:
        s3_uri (str): The S3 URI of the folder to download (e.g., "s3://bucket-name/path/to/folder/").
        local_directory (str): The local directory where the contents will be downloaded.

    Returns:
        None
    """

    # Parse the S3 URI
    s3_bucket, s3_path = parse_s3_uri(s3_uri)

    # Initialize the S3 client
    s3_client = boto3.client("s3")

    # List objects in the S3 folder
    response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=s3_path)

    # Download each JPG file in the folder
    for obj in response.get("Contents", []):
        s3_key = obj["Key"]

        # Check if the object is a JPG file
        if s3_key.lower().endswith(".jpg"):
            local_file = os.path.join(local_directory, os.path.basename(s3_key))
            os.makedirs(os.path.dirname(local_file), exist_ok=True)
            s3_client.download_file(s3_bucket, s3_key, local_file)

def parse_s3_uri(s3_uri):
    """
    Parse an S3 URI and extract the bucket and path.

    Parameters:
        s3_uri (str): The S3 URI (e.g., "s3://bucket-name/path/to/folder/").

    Returns:
        (str, str): A tuple containing the bucket name and the path.
    """

    s3_uri = s3_uri.strip("s3://")
    parts = s3_uri.split("/", 1)
    s3_bucket = parts[0]
    s3_path = parts[1] if len(parts) == 2 else ""

    return s3_bucket, s3_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3_uri", type=str, required=True)
    parser.add_argument("--local_dir", type=str, required=True)
    args = parser.parse_args()
    download_s3_folder(args.s3_uri, args.local_dir)
