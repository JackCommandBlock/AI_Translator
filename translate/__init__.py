"""翻译与术语模块。"""
from .translator import Translator, build_client
from .glossary import load_glossary, apply_glossary

__all__ = ["Translator", "build_client", "load_glossary", "apply_glossary"]

