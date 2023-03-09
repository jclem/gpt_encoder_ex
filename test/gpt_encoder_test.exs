defmodule GPTEncoderTest do
  use ExUnit.Case
  doctest GPTEncoder

  test "greets the world" do
    assert GPTEncoder.hello() == :world
  end
end
