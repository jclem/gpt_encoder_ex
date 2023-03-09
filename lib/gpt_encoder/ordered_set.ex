defmodule GPTEncoder.OrderedSet do
  @moduledoc false

  # Internal-only ordered set implementation.

  @type t(v) :: list(v)

  @doc "Create a new ordered set from an enum."
  @spec new(list(v)) :: t(v) when v: any
  def new(list), do: Enum.reduce(list, [], &add(&2, &1))

  @doc "Add a value to an ordered set."
  @spec add(t(v), v) :: t(v) when v: any
  def add(set, val), do: if(Enum.member?(set, val), do: set, else: set ++ [val])
end
