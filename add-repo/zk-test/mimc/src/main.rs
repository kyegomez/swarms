mod blind;
mod pre;
mod pedersen;
// mod non_blind;

// 引入所需的库
use ark_bn254::{Bn254, Fr, G1Projective,G1Affine};
use ark_ff::{BigInteger256, FftField, Field, PrimeField, UniformRand};
use ark_serialize::CanonicalSerialize;
use rand_core::OsRng;
use sha2::{Sha256, Digest};
use ark_ec::{ProjectiveCurve, AffineCurve};

use std::time::{Instant, Duration};

use pedersen::*;


fn main() {
 
    let starttime = Instant::now();

    //2跳的代理重加密
   // let _ =pre::run();

    let _ =blind::run();

    let duration =starttime.elapsed();
    let elapsed_ms = duration.as_millis();

    println!("Total time: {} ms\n", elapsed_ms);

    let k = Fr::rand(&mut OsRng);
    let starttime = Instant::now();
    let test_exp = Fr::multiplicative_generator().pow(k.into_repr());
    let duration =starttime.elapsed();
    let elapsed_us = duration.as_secs_f64() *1000.0*1000.0;
    println!("test_exp time: {} us\n", elapsed_us);

    let starttime = Instant::now();
    let test_mul = test_exp*k;
    let duration =starttime.elapsed();
    let elapsed_us = duration.as_secs_f64() *1000.0*1000.0;
    println!("test_mul: {} us\n", elapsed_us);

    let message: &[u8] =b"aasdiahsduoashdzxkhcvijashdasdas";

let starttime = Instant::now();
    let hash = Sha256::digest(message);
    let mut wide_hash = [0_u8; 64];
    wide_hash[..32].copy_from_slice(&hash);
    
    let duration =starttime.elapsed();
    let elapsed_ms = duration.as_secs_f64() *1000.0*1000.0;
    println!("test_h: {} us\n", elapsed_ms);
    
    //println!("Result of g * h(r): {:#?}", result);
}