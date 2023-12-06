

def to_camel_case(s: str) -> str:
    t = ""
    for i in range(len(s) - 1):
        if s[i].isupper() and s[i + 1].islower():
            t += "_"
        t += s[i].lower()
    return t + s[-1]
