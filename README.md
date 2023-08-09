# LLAMA2 Rust port

Simple dumb llama2.c by Andrej Karpathy. Achieves about 70tokens/sec on my 8-year old Intel i5-6200U CPU, so lots of room for improvement

To run
```
cargo run --release -- --prompt "Hello, LLAMA2! How do you do?" --checkpoint-file <path-to-file> --temperature 1.0
```
