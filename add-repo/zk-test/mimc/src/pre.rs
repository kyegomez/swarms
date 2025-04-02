use recrypt::prelude::*;
use std::time::{Instant, Duration};
//实现了2跳的代理重加密
pub(crate) fn run() -> Result<(),Box<dyn std::error::Error>> {
    // 初始化Recrypt
    let recrypt = Recrypt::new();

    //生成代理商的signing keys
    let signing_keypair= recrypt.generate_ed25519_key_pair();

    // 生成用户的密钥对
    let (private_key_data_owner, public_key_data_owner) = recrypt.generate_key_pair().unwrap();
    let (private_key_manage_center, public_key_manage_center) = recrypt.generate_key_pair().unwrap();
    let (private_key_data_user, public_key_data_user) = recrypt.generate_key_pair().unwrap();

    
    let plaintext = recrypt::api::Plaintext::new([23_u8;384]);

    // 加密数据 其中使用了代理商的公钥
    let encrypted_data = recrypt.encrypt(&plaintext, &public_key_data_owner, &signing_keypair).unwrap();
 

    // 生成代理重加密密钥
    //从a到b
    let re_encryption_key_do_to_mc = recrypt.generate_transform_key(
        &private_key_data_owner, 
        &public_key_manage_center, 
        &signing_keypair).unwrap();

        
        
        //从b到c
    let re_encryption_key_mc_to_du = recrypt.generate_transform_key(
        &private_key_manage_center, 
        &public_key_data_user, 
        &signing_keypair).unwrap();


    // 重加密流程
    let encrypted_data_mc = recrypt.transform(
        encrypted_data, 
        re_encryption_key_do_to_mc, 
        &signing_keypair).unwrap();

        

    //重加密密文 b to c
    let encrypted_data_du = recrypt.transform(
        encrypted_data_mc, 
        re_encryption_key_mc_to_du, 
        &signing_keypair).unwrap();

    // 解密数据
    let starttime = Instant::now();
    let decrypted_data = recrypt.decrypt(encrypted_data_du, &private_key_data_user).unwrap();

    let duration =starttime.elapsed();
        let elapsed_ms = duration.as_millis();
    
        println!("解密时间: {} ms\n", elapsed_ms);
    assert_eq!(decrypted_data, plaintext);
    println!("成功解密数据！");
    Ok(())
}
