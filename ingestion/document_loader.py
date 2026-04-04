import pandas as pd
from pypdf import PdfReader

def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text()

    return text


def load_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def load_csv(file_path):
    df = pd.read_csv(file_path)
    return df.to_string()


def load_excel(file_path):
    df = pd.read_excel(file_path)
    return df.to_string()


def load_document(file_path):

    if file_path.endswith(".pdf"):
        return load_pdf(file_path)

    elif file_path.endswith(".txt"):
        return load_txt(file_path)

    elif file_path.endswith(".csv"):
        return load_csv(file_path)

    elif file_path.endswith(".xlsx"):
        return load_excel(file_path)

    else:
        raise ValueError("Unsupported file format")