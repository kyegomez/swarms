import requests 

def download_weights_from_url(url: str, save_path: str = "models/weights.pth"):
    """
    Downloads model weights from the given URL and saves them to the specified path.

    Args:
        url (str): The URL from which to download the model weights.
        save_path (str, optional): The path where the downloaded weights should be saved. 
            Defaults to "models/weights.pth".
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
        
    print(f"Model weights downloaded and saved to {save_path}")