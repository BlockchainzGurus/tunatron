//! Stampa il calldata di `attest(cid, reportHash, modelHash)`: utile per `cast send` e per verificare l'ABI.
//!
//!   cargo run --example encode_attest -- <cid> <reportHash 0x..64 hex> <modelHash 0x..64 hex>

use sast_engine::notarizer::encode_attest_call;

fn hash_arg(s: &str) -> [u8; 32] {
    let bytes = hex::decode(s.trim_start_matches("0x")).expect("hash esadecimale non valido");
    bytes.try_into().expect("l'hash deve essere di 32 byte")
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [cid, report, model] = args.as_slice() else {
        eprintln!("uso: encode_attest <cid> <reportHash> <modelHash>");
        std::process::exit(2);
    };
    println!("0x{}", hex::encode(encode_attest_call(cid, hash_arg(report), hash_arg(model))));
}
