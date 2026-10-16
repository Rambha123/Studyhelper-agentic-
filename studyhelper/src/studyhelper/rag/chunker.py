def chunk_text(text: str, size: int = 600, overlap: int = 80) -> list[str]:
    """Split text into ~size-char chunks with overlap, snapping to word boundaries."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = text.split()
    chunks, current = [], []
    length = 0
    for w in words:
        if length + len(w) + 1 > size and current:
            chunks.append(" ".join(current))
            # carry over the tail words that fit in the overlap window
            tail, tl = [], 0
            for x in reversed(current):
                if tl + len(x) + 1 > overlap:
                    break
                tail.insert(0, x)
                tl += len(x) + 1
            current, length = tail, tl
        current.append(w)
        length += len(w) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks
