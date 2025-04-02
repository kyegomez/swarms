//use bls12_381::*;
//use ff::*;
use rand_core::OsRng;

use ark_bn254::{Bn254, Fr, g1, G1Affine, G1Projective, FrParameters};
use ark_ec::{ ProjectiveCurve, AffineCurve};
use ark_ff::{UniformRand, Fp256};
use ark_ff::{Field, PrimeField,BigInteger256};

use ark_serialize::*;
use num_bigint::BigUint;



// 基于bls12-381仿射映射的pedersen承诺
/*fn pedersen_commit(value: Scalar, blinding_factor: Scalar) -> G1Affine {
       //生成曲线上的基点
    let G = G1Affine::generator();
    let H = G1Affine::generator();

    (G * value + H * blinding_factor).into()
}*/

//基于bn254仿射映射的pedersen承诺
fn pedersen_commit(value: Fr, blinding_factor: Fr) -> G1Affine {
    // Generator points in affine coordinates
    let g_affine = G1Affine::prime_subgroup_generator();
    let h_affine = G1Affine::from(G1Projective::rand(&mut OsRng));
   

    println!("g的值{:?}", g_affine);
    println!("h的值{:?}",h_affine);
    

    // Compute the Pedersen commitment
    (g_affine.mul(value) + h_affine.mul(blinding_factor)).into()
}

pub(crate) fn run() -> Result<(), Box<dyn std::error::Error>> {
 

    //生成随机数
    let blind = Fr::new(10.into());

    //承诺的消息
    let message: Fp256<FrParameters> = Fr::new(3.into());
    //计算perdersen承诺
    let commitment = pedersen_commit(message, blind);

    println!("承诺的消息是{:#?}\n 选择的随机值是{:#?}\npedersen承诺为:{:#?}",message,blind,commitment);

    let x = commitment.x;
    let y = commitment.y;

    let mut commit_compressed_bytes = Vec::new();
    let mut message_bytes = Vec::new();
   

    //字节化Fp256变为十六进制
    commitment.serialize(&mut commit_compressed_bytes).unwrap();
    message.serialize(&mut message_bytes).unwrap();
    
    


    // 将字节转换为十进制
    let messgage_dem = BigUint::from_bytes_be(&message_bytes);

    // 打印十进制坐标
    println!("commit_compressed_bytes is: {:?}", commit_compressed_bytes);
    println!("The length of commit_compressed_bytes is : {}",commit_compressed_bytes.len() );
    println!("message_bytes is: {:?}", message_bytes);
    println!("message转换为uint256后的值:{:#?}", messgage_dem);



    
Ok(())
}
