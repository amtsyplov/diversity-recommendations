def to_camel_case(s: str) -> str:
    t = s[0].lower()
    for i in range(1, len(s) - 1):
        if s[i].isupper() and s[i + 1].islower():
            t += "_"
        t += s[i].lower()
    return t + s[-1].lower()
