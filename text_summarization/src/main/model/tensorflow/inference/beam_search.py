

from collections import namedtuple

class BeamSearchConfig(
    namedtuple("BeamSearchConfig", ["beam_width", "length_penalty_weight", "start_token", "end_token",
                                    "embed_matrix", "vocab_size", "summ_length", "max_length"])):
  """Configuration object for beam search.

  Args:
    beam_width: Number of beams to use, an integer
    vocab_size: Output vocabulary size
    eos_token: The id of the EOS token, used to mark beams as "done"
    length_penalty_weight: Weight for the length penalty factor. 0.0 disables
      the penalty.
    choose_successors_fn: A function used to choose beam successors based
      on their scores. Maps from (scores, config) => (chosen scores, chosen_ids)
  """
  pass