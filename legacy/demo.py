from langdetect import detect

def is_portuguese(question):
    try:
        # Detecta o idioma do texto
        language = detect(question)
        # Retorna True se o idioma for português ('pt'), False caso contrário
        return language == 'en'
        # return language == 'pt'
    except:
        # Retorna False se houver erro na detecção (ex.: texto muito curto)
        return False

# Exemplo de uso
pergunta = "Qual é o capital do Brasil?"
print(is_portuguese(pergunta))  # True

pergunta = "What is the capital of Brazil?"
print(is_portuguese(pergunta))  # False

from deep_translator import GoogleTranslator
from langdetect import detect

def translate_to_english(text):
    try:
        # Detecta o idioma do texto
        language = detect(text)
        # Verifica se o idioma é português
        if language == 'pt':
            # Traduz do português para o inglês
            translator = GoogleTranslator(source='pt', target='en')
            translated_text = translator.translate(text)
            return translated_text
        else:
            return "Erro: O texto de entrada não está em português."
    except:
        return "Erro: Não foi possível detectar o idioma ou traduzir o texto."

def translate_to_portuguese(text):
    try:
        # Detecta o idioma do texto
        language = detect(text)
        # Verifica se o idioma é inglês
        if language == 'en':
            # Traduz do inglês para o português
            translator = GoogleTranslator(source='en', target='pt')
            translated_text = translator.translate(text)
            return translated_text
        else:
            return "Erro: O texto de entrada não está em inglês."
    except:
        return "Erro: Não foi possível detectar o idioma ou traduzir o texto."

# Exemplo de uso
texto_pt = "Qual é a capital do Brasil?"
print("Tradução para o inglês:", translate_to_english(texto_pt))  # What is the capital of Brazil?

texto_en = "What is the capital of Brazil?"
print("Tradução para o português:", translate_to_portuguese(texto_en))  # Qual é a capital do Brasil?