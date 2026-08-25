import webbrowser
import requests
from urllib.parse import urlparse


def download_file(url):
    try:
        parsed_url = urlparse(url)
        response = requests.get(url)
        if response.status_code == 200:
            file_name = parsed_url.path.split('/')[-1]
            with open(file_name, 'wb') as file:
                file.write(response.content)
            print(f'File "{file_name}" downloaded successfully.')
        else:
            print(f'Failed to download file. Status code: {response.status_code}')
    except Exception as e:
        print(f'An error occurred: {e}')


def open_link(url):
    webbrowser.open(url)


url = 'https://bureauveritas.sharepoint.com/:x:/r/teams/InnovationGenAI/_layouts/15/Doc.aspx?sourcedoc=%7BA275C990-E383-4743-A164-4BD909E325F0%7D&file=ooo.xlsx'
download_file(url)
open_link(url)