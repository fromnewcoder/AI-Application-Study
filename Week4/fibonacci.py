a, b = 0, 1; [print(a, end=' ') for _ in iter(int, 1) if (a := (b, b := a + b)[0]) <= 100 or exit()]
