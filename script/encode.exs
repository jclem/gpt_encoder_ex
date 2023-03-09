GPTEncoder.start_link()

input = System.argv() |> List.first()

input
|> GPTEncoder.encode()
|> dbg
