


class EncoderOutput():
    """Encodes the input as embeddings."""

    def __init__(self, last_hidden_state, pooler_output):
        self.last_hidden_state=last_hidden_state
        self.pooler_output=pooler_output

