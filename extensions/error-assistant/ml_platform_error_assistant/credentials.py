import keyring


KEYRING_SERVICE = "ModelMind Notebook"
GEMINI_KEY_NAME = "Gemini API Key"


def get_stored_gemini_api_key():
    return keyring.get_password(KEYRING_SERVICE, GEMINI_KEY_NAME)


def store_gemini_api_key(api_key):
    keyring.set_password(KEYRING_SERVICE, GEMINI_KEY_NAME, api_key)


def delete_stored_gemini_api_key():
    try:
        keyring.delete_password(KEYRING_SERVICE, GEMINI_KEY_NAME)
    except keyring.errors.PasswordDeleteError:
        pass
