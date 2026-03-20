"""Command-line interface for markovonnx."""

import argparse
import sys

from markovonnx.version import __version__


def cmd_train(args: argparse.Namespace) -> None:
    """Train a Markov chain and save as .markov archive."""
    from markovonnx.archive import save_markov_archive
    from markovonnx.markov import MarkovChain
    from markovonnx.tokenizers import char_tokenize, corpus_iter, get_tokenize_fn, word_tokenize
    from markovonnx.vocabulary import Vocabulary

    tokenize_fn = get_tokenize_fn(args.mode)

    print(f"Building vocabulary from {args.corpus}...")
    vocab = Vocabulary(max_vocab=args.max_vocab)
    vocab.build_streaming(args.corpus, tokenize_fn=tokenize_fn, max_lines=args.max_lines)
    print(f"Vocabulary: {vocab.size} tokens")

    print(f"Training MarkovChain(order={args.order}, backoff={args.backoff})...")
    mc = MarkovChain(
        order=args.order,
        vocab=vocab,
        smoothing=args.smoothing,
        backoff=args.backoff,
    )
    mc.fit_streaming(args.corpus, tokenize_fn=tokenize_fn, max_lines=args.max_lines)

    save_markov_archive(mc, args.output)

    export_c = getattr(args, "export_c", "")
    if export_c:
        from markovonnx.c_export import export_markov_c_header
        no_quantize = getattr(args, "no_quantize", False)
        progmem = getattr(args, "progmem", False)
        export_markov_c_header(mc, export_c, quantize=not no_quantize, progmem=progmem)
        print(f"C header written: {export_c}")


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate text from a .markov archive."""
    from markovonnx.archive import load_markov_archive
    from markovonnx.generate import generate_markov

    loaded = load_markov_archive(args.archive)
    rt = loaded["runtime"]
    config = loaded["config"]
    order = config.get("order", 2)
    mode = args.mode

    text = generate_markov(
        rt, args.seed, args.length,
        temperature=args.temperature, mode=mode, order=order,
    )
    print(text)


def cmd_info(args: argparse.Namespace) -> None:
    """Show metadata from a .markov archive."""
    import json
    import zipfile
    from pathlib import Path

    with zipfile.ZipFile(args.archive, "r") as zf:
        config = json.loads(zf.read("config.json"))

    size_kb = Path(args.archive).stat().st_size / 1024
    print(f"Archive:    {args.archive} ({size_kb:.1f} KB)")
    for key, val in config.items():
        print(f"  {key}: {val}")


def main() -> None:
    """Entry point for ``markovonnx`` CLI."""
    parser = argparse.ArgumentParser(
        prog="markovonnx",
        description="Markov chains and HMMs with ONNX export/inference",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    # -- train ----------------------------------------------------------------
    p_train = sub.add_parser("train", help="Train a Markov chain from a text corpus")
    p_train.add_argument("corpus", help="Path to text corpus (one line per sequence)")
    p_train.add_argument("-o", "--output", default="model.markov", help="Output .markov archive path")
    p_train.add_argument("--mode", choices=["char", "word"], default="char", help="Tokenization mode")
    p_train.add_argument("--order", type=int, default=3, help="N-gram order")
    p_train.add_argument("--smoothing", type=float, default=1e-5, help="Laplace smoothing alpha")
    p_train.add_argument("--max-vocab", type=int, default=0, help="Max vocabulary size (0=unlimited)")
    p_train.add_argument("--max-lines", type=int, default=0, help="Max corpus lines (0=unlimited)")
    p_train.add_argument("--backoff", action="store_true", help="Enable interpolated backoff")
    p_train.add_argument("--export-c", metavar="PATH", default="", help="Also export a C header (.h) for embedded/ESP32 use")
    p_train.add_argument("--no-quantize", action="store_true", help="Use float32 instead of uint8 in C header (larger)")
    p_train.add_argument("--progmem", action="store_true", help="Annotate C arrays with ESP32 .rodata section attribute")

    # -- generate -------------------------------------------------------------
    p_gen = sub.add_parser("generate", help="Generate text from a .markov archive")
    p_gen.add_argument("archive", help="Path to .markov archive")
    p_gen.add_argument("--seed", default="", help="Seed text")
    p_gen.add_argument("--length", type=int, default=100, help="Tokens to generate")
    p_gen.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    p_gen.add_argument("--mode", choices=["char", "word"], default="char", help="Tokenization mode")

    # -- info -----------------------------------------------------------------
    p_info = sub.add_parser("info", help="Show metadata from a .markov archive")
    p_info.add_argument("archive", help="Path to .markov archive")

    args = parser.parse_args()
    if args.command == "train":
        cmd_train(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "info":
        cmd_info(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
