defmodule GPTEncoder do
  @moduledoc """
  A BPE encoder for GPT-x large language models.

  This turns a string into a list of integers representing tokens. This list can
  be used as input to GPT-x models, but it's also useful in predicting token
  consumption when using GPT APIs that accept text.
  """

  use GenServer

  @pat ~r/'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/u

  defmodule State do
    @moduledoc false

    defstruct [:byte_encoder, :bpe_ranks, :encoder, :cache]

    @type t :: %__MODULE__{
            byte_encoder: map,
            bpe_ranks: map,
            encoder: map,
            cache: map
          }
  end

  # Client API

  @doc """
  Start the encoder server.
  """
  @spec start_link(keyword, keyword) :: GenServer.on_start()
  def start_link(init_opts \\ [], server_opts \\ [name: __MODULE__]) do
    GenServer.start_link(__MODULE__, init_opts, server_opts)
  end

  @doc """
  Encode the given string.
  """
  @spec encode(String.t()) :: list(integer)
  def encode(string) do
    GenServer.call(__MODULE__, {:encode, string})
  end

  @doc """
  Encode the given string using the encoder `pid`.
  """
  @spec encode(pid, String.t()) :: list(integer)
  def encode(pid, string) do
    GenServer.call(pid, {:encode, string})
  end

  # GenServer callbacks

  @impl true
  @spec init(keyword) :: {:ok, State.t()}
  def init(opts) do
    state = initialize(opts)
    {:ok, state}
  end

  @impl true
  @spec handle_call({:encode, String.t()}, GenServer.from(), State.t()) ::
          {:reply, list(integer), State.t()}
  def handle_call({:encode, text}, _from, %State{} = state) do
    {encoded, new_state} = do_encode(text, state)
    {:reply, encoded, new_state}
  end

  # Internal API

  @spec initialize(keyword) :: any
  defp initialize(opts) do
    support_path = :code.priv_dir(:gpt_encoder)
    bpe_path = Keyword.get(opts, :bpe_path, Path.join(support_path, "vocab.bpe"))
    encoder_path = Keyword.get(opts, :encoder_path, Path.join(support_path, "encoder.json"))

    bpe_ranks =
      bpe_path
      |> File.read!()
      |> String.split("\n")
      |> Enum.slice(1..-2)
      |> Enum.map(&String.split/1)
      |> Enum.map(&List.to_tuple/1)
      |> Enum.with_index()
      |> Map.new()

    encoder =
      encoder_path
      |> File.read!()
      |> Jason.decode!()

    %State{
      byte_encoder: bytes_to_unicode(),
      bpe_ranks: bpe_ranks,
      encoder: encoder,
      cache: %{}
    }
  end

  @spec bytes_to_unicode() :: map
  defp bytes_to_unicode do
    # A map of utf-8 bytes to strings containing the unicode character.
    # https://github.com/openai/gpt-2/blob/a74da5d99abaaba920de8131d64da2862a8f213b/src/encoder.py#L9-L28
    codepoints = Enum.concat([?!..?~, ?¡..?¬, ?®..?ÿ])
    extras = for c <- 0..(2 ** 8 - 1), !Enum.member?(codepoints, c), do: c

    codepoints
    |> Enum.concat(extras)
    |> Enum.zip(
      codepoints
      |> Enum.concat(Enum.with_index(extras, fn _c, i -> 2 ** 8 + i end))
      |> Enum.map(&to_string([&1]))
    )
    |> Map.new()
  end

  @spec do_encode(String.t(), State.t()) :: {list(integer), State.t()}
  defp do_encode(text, %State{} = state) do
    {words, state} =
      @pat
      |> Regex.scan(text)
      |> List.flatten()
      |> Enum.map(fn token ->
        token
        |> :binary.bin_to_list()
        |> Enum.map(&Map.get(state.byte_encoder, &1))
        |> Enum.join()
      end)
      |> Enum.reduce({[], state}, fn word, {words, state} ->
        {word, state} = bpe(word, state)
        {[word | words], state}
      end)

    {words
     |> Enum.reverse()
     |> Enum.map(fn token ->
       token
       |> String.split(" ")
       |> Enum.map(&Map.get(state.encoder, &1))
     end)
     |> List.flatten(), state}
  end

  defp bpe(token, %State{} = state) do
    if cached = Map.get(state.cache, token) do
      {cached, state}
    else
      word = token |> to_charlist |> Enum.chunk_every(1)
      max_rank = Enum.count(state.bpe_ranks)

      word
      |> flatten_words(max_rank, state)
      |> Enum.join(" ")
      |> then(&{&1, %State{state | cache: Map.put(state.cache, token, &1)}})
    end
  end

  defp flatten_words(word, _, _) when length(word) == 1 do
    word
  end

  defp flatten_words(word, max_rank, vocab) do
    pairs = get_pairs(word)
    best_ranked = Enum.min_by(Enum.reverse(pairs), &Map.get(vocab.bpe_ranks, &1, max_rank))
    rank = Map.get(vocab.bpe_ranks, best_ranked, max_rank)

    if rank == max_rank do
      word
    else
      {first, second} = best_ranked
      first = to_charlist(first)
      second = to_charlist(second)

      new_word =
        Stream.cycle([true])
        |> Enum.reduce_while({[], 0}, fn _, {acc, i} ->
          if i >= length(word) do
            {:halt, {acc, i}}
          else
            word
            |> Enum.with_index()
            |> Enum.find_index(fn {char, index} ->
              if index < i do
                nil
              else
                char == first
              end
            end)
            |> case do
              nil ->
                acc = acc ++ Enum.slice(word, i..-1)
                {:halt, {acc, i}}

              j ->
                acc =
                  if j == 0 do
                    ''
                  else
                    acc ++ Enum.slice(word, i..(j - 1))
                  end

                i = j

                {acc, i} =
                  if Enum.at(word, i) == first and i < length(word) - 1 and
                       Enum.at(word, i + 1) == second do
                    {acc ++ [first ++ second], i + 2}
                  else
                    {acc ++ [Enum.at(word, i)], i + 1}
                  end

                {:cont, {acc, i}}
            end
          end
        end)
        |> elem(0)

      flatten_words(new_word, max_rank, vocab)
    end
  end

  def merge_word(selected, [a | [b | rest]], acc) when {a, b} == selected do
    merge_word(selected, rest, [a <> b | acc])
  end

  def merge_word(selected, [a | [b | rest]], acc) do
    merge_word(selected, [b | rest], [a | acc])
  end

  def merge_word(selected, [a | []], acc) do
    merge_word(selected, [], [a | acc])
  end

  def merge_word(_selected, [], acc), do: Enum.reverse(acc)

  def get_pairs(word) do
    word
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.map(&tuple_pair/1)
    |> GPTEncoder.OrderedSet.new()
  end

  defp tuple_pair(pair) do
    pair
    |> Enum.map(&List.to_string([&1]))
    |> List.to_tuple()
  end
end
