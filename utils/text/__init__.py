""" from https://github.com/keithito/tacotron """
from . import kana

from . import sampa


def sampa_to_sequence(inputs):
    return [sampa.sampa_to_id[x] for x in inputs]


def text_to_sequence(text):
    """Converts a string of text to a sequence of IDs corresponding to the symbo
ls in the text.

    Args:
      text: string to convert to a sequence

    Returns:
      List of integers corresponding to the symbols in the text
    """

    if isinstance(text, str):
        kn = kana.KanaSent()
        kn.set_jeitaKana(text)
        inputs = kn.get_input_feature()
    else:
        inputs = text

    return sampa_to_sequence(inputs)

