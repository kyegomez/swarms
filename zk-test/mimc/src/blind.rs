use ark_bn254::{Fr, G1Projective, G2Projective, Bn254};
use ark_ec::{PairingEngine, ProjectiveCurve};
use ark_ff::{Field, UniformRand, PrimeField};
use rand_core::OsRng;
use sha2::{Digest, Sha256};
use std::time::Instant;

//使用bn254来实现了基于bls的盲签名算法 验证使用了paring来进行验证

// 生成密钥对
fn generate_keys() -> (Fr, G1Projective) {
    let private_key = Fr::rand(&mut OsRng);
    let public_key = G1Projective::prime_subgroup_generator().mul(private_key.into_repr());

    (private_key, public_key)
}

// 对消息进行盲化
fn blind_message(message: &[u8], blind_factor: &Fr) -> G2Projective {
    let hash = Sha256::digest(message);
    let mut wide_hash = [0_u8; 64];
    wide_hash[..32].copy_from_slice(&hash);
    
    let hash_scalar = Fr::from_be_bytes_mod_order(&wide_hash);
    
    G2Projective::prime_subgroup_generator().mul(hash_scalar.into_repr()).mul(blind_factor.into_repr())
}

// 签名盲化消息
fn sign_blinded_message(private_key: &Fr, blinded_message: &G2Projective) -> G2Projective {
    blinded_message.mul(private_key.into_repr())
}

// 去盲化签名
fn unblind_signature(blinded_signature: &G2Projective, blind_factor: &Fr) -> G2Projective {
    let blind_factor_inv = blind_factor.inverse().unwrap();  //计算盲化因子的逆
    blinded_signature.mul(blind_factor_inv.into_repr()) //返回一个去盲化的签名
}

// 验证签名
fn verify_signature(public_key: &G1Projective, message: &[u8], signature: &G2Projective) -> bool {
    let hash = Sha256::digest(message);
    let mut wide_hash = [0_u8; 64];
    wide_hash[..32].copy_from_slice(&hash);

   
    
    let hash_scalar = Fr::from_be_bytes_mod_order(&wide_hash);
    let g1 = G1Projective::prime_subgroup_generator();
    let starttime = Instant::now();
    let sig_g1_pairing = Bn254::pairing(g1.into_affine(), signature.into_affine());
    let duration = starttime.elapsed();
    let elapsed_ms = duration.as_secs_f64() *1000.0;
    println!("Test paring: {} ms", elapsed_ms);

    let hash_pk_pairing = Bn254::pairing(public_key.into_affine(), (G2Projective::prime_subgroup_generator().mul(hash_scalar.into_repr())).into_affine());
  // let hash_pk_pairing = Bn254::pairing(public_key.into_affine(), (G2Projective::prime_subgroup_generator()).into_affine()).cyclotomic_exp(hash_scalar.into_repr());

    sig_g1_pairing == hash_pk_pairing
}
//验证聚合签名
// fn verify_bls_signature(public_key: &[G1Projective], message: &[u8], signature: &G2Projective) -> bool {

    
//     let hash = Sha256::digest(message);
//     let mut wide_hash = [0_u8; 64];
//     wide_hash[..32].copy_from_slice(&hash);

//     let mut key: G1Projective = Default::default();

//     //计算公钥和
//     for i in public_key.iter() {
//         key += i;
//     }
    
//     let hash_scalar = Fr::from_be_bytes_mod_order(&wide_hash);
//     let g1 = G1Projective::prime_subgroup_generator();
//     let sig_g1_pairing = Bn254::pairing(g1.into_affine(), signature.into_affine());
//     let hash_pk_pairing = Bn254::pairing(key.into_affine(), (G2Projective::prime_subgroup_generator().mul(hash_scalar.into_repr())).into_affine());

//     sig_g1_pairing == hash_pk_pairing

// }

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    
    // ...
    let (private_key, public_key) = generate_keys();
    let message = b"Hello, Blind BLSBLSBLSBLSBLSBLS!";
    let blind_factor = Fr::rand(&mut OsRng);

    println!("明文:{:?}\n 盲化因子:{}",message,blind_factor);
    let starttime = Instant::now();
    let blinded_message = blind_message(message, &blind_factor);
    
    let blinded_signature = sign_blinded_message(&private_key, &blinded_message);
   
    let signature = unblind_signature(&blinded_signature, &blind_factor);

    let duration = starttime.elapsed();
    let elapsed_ms = duration.as_secs_f64() *1000.0;
    println!("Sign time: {} ms", elapsed_ms);
    let starttime = Instant::now();
    let is_valid = verify_signature(&public_key, message, &signature);
    let duration = starttime.elapsed();
    let elapsed_ms = duration.as_secs_f64() *1000.0;
    println!("Verify time: {} ms", elapsed_ms);
    println!("Signature is valid: {}", is_valid);
    Ok(())
}
