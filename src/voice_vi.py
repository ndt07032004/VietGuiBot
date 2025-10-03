# voice_vi.py
from enum import Enum

class EnumVoice(Enum):
    HOAI_MY = {"name": "Hoai My", "voiceName": "HoaiMy", "styleList": {"neutral": "general"}}
    NAM_MINH = {"name": "Nam Minh", "voiceName": "NamMinh", "styleList": {"neutral": "general"}}

def get_voice_list():
    return [EnumVoice.HOAI_MY, EnumVoice.NAM_MINH]

def get_voice_of(name):
    for voice in get_voice_list():
        if voice.value["name"] == name:
            return voice
    return None
