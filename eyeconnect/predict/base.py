"""Частотный предикт MVP: префиксные подсказки top-3 (дока 5.2-5.3 упрощённо).

Полноценные kenlm/Presage — фаза 3. Здесь встроенные мини-словари EN/KZ
+ автозагрузка data/kz_top5k.txt и data/en_top10k.txt (по слову на строку).
"""
from pathlib import Path

_EN_MINI = """the be to of and a in that have hello how are you what where when with for this from
they say her she will one all would there their eye type gaze keyboard help thanks please""".split()

_KZ_MINI = """және бұл деп сәлем қалайсың рахмет көмек үшін мен сен біз сіз олар бар жоқ
керек болады еді аты жөні мектеп компьютер көз перне сөз сөйлеу""".split()

_KZ_SUFFIX = ["лар", "лер", "дар", "дер", "ның", "нің", "ға", "ге", "ды", "ді"]


class PrefixPredictor:
    def __init__(self, lang="KZ"):
        self.lang = lang
        self.words = list(_KZ_MINI if lang == "KZ" else _EN_MINI)
        self._load_external(lang)

    def _load_external(self, lang):
        base = Path(__file__).resolve().parents[1] / "data"
        fn = base / ("kz_top5k.txt" if lang == "KZ" else "en_top10k.txt")
        if fn.exists():
            extra = [w.strip() for w in fn.read_text(encoding="utf-8").splitlines() if w.strip()]
            # внешние выше приоритетом
            self.words = extra + self.words

    def suggest(self, prefix: str, n=3):
        p = (prefix or "").strip().split()[-1] if (prefix or "").strip() else ""
        if not p:
            return self.words[:n]
        pl = p.lower()
        out = [w for w in self.words if w.lower().startswith(pl) and w.lower() != pl]
        # KZ: если слово набрано целиком — предложить суффиксы (агглютинация)
        if self.lang == "KZ" and len(out) < n and p:
            out += [p + s for s in _KZ_SUFFIX]
        return out[:n]
