from deep_translator import GoogleTranslator
from langdetect import detect

from src.log.log import logger

def is_portuguese(text):
    try:
        # Detects the language of the text
        language = detect(text)
        # Returns True if the language is Portuguese ('pt'), False otherwise
        return language == 'pt'
    except:
        # Returns False if there is an error in detection (e.g., text too short)
        return False

def is_english(text):
    try:
        # Detects the language of the text
        language = detect(text)
        # Returns True if the language is English ('en'), False otherwise
        return language == 'en'
    except:
        # Returns False if there is an error in detection (e.g., text too short)
        return False

def translate_to_english(text):
    try:
        # Checks if the language is Portuguese
        if is_portuguese(text):
            # Translates from Portuguese to English
            translator = GoogleTranslator(source='pt', target='en')
            translated_text = translator.translate(text)
            logger.info("Translate text to English successfully.")
            return translated_text
        else:
            logger.error(f"Error: The input text is not in Portuguese.")
            return "Error: The input text is not in Portuguese."
    except:
        logger.error(f"Error: Unable to detect the language or translate the text.")
        return "Error: Unable to detect the language or translate the text."

def translate_to_portuguese(text):
    try:
        # Checks if the language is English
        if is_english(text):
            # Translates from English to Portuguese
            translator = GoogleTranslator(source='en', target='pt')
            translated_text = translator.translate(text)
            logger.info("Translate text to Portuguese successfully.")
            return translated_text
        else:
            logger.error(f"Error: The input text is not in English.")
            return "Error: The input text is not in English."
    except:
        logger.error(f"Error: Unable to detect the language or translate the text.")
        return "Error: Unable to detect the language or translate the text."
