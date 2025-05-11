from googletrans import Translator
import asyncio



translator = Translator()

async def nltkdata(text):
    translator = Translator()
    translated = await translator.translate(text, src='ta', dest='en')
    return translated.text


def nltkdataRun(text):
    return asyncio.run(nltkdata(text=text))
