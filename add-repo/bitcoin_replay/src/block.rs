use std::time::{SystemTime, UNIX_EPOCH};
use sha2::{Sha256, Digest};


use crate::{utils, proofofwork::ProofOfWork};

pub struct Block {
    //创建区块的时间戳
    pub time_stamp: i64,
    //存储区块的信息
    pub data: Vec<u8>,
    //上一区块的hash
    pub pre_block_hash: Vec<u8>,
    //当前区块的hash
    pub hash :Vec<u8>,
    //挖矿目标值
    pub nonce: u32,
}

impl Block {
    pub(crate) fn new_genesis_block() -> Block {
        let  pre_block_hash = vec![];
        Block::new_block_chain("Genesis Block", pre_block_hash)

    }

    pub fn new_block_chain(data :&str, pre_block_hash: Vec<u8>) ->Block {
        let mut block = Block {
            time_stamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64,
            data: data.as_bytes().to_vec(),
            pre_block_hash,
            hash: Vec::new(),
            nonce: 0,
        };

        //block.set_hash();
        //使用pow方法生成区块的hash值
        let pow = ProofOfWork::new(&block);
        let (nonce, hash) = pow.run();
        block.nonce = nonce;
        block.hash = hash;
        block

    }

    //设置区块的hash值，使用当前的时间戳 上一个区块的hash值，以及区块交易数据来进行创建
    pub fn set_hash(&mut self) {
        //concat change from "hello" "world" to "helloworld"
        let headers: Vec<u8> = vec![&self.time_stamp.to_le_bytes() as &[u8],
        &self.data,
        &self.pre_block_hash
        ].concat();
        

       

        let mut hasher :Sha256 =Sha256::new();
        hasher.update(headers);
        
        

        self.hash = hasher.finalize().to_vec();


    }

    pub fn print_content(&self) {
        println!("Timestamp:{}",self.time_stamp);
        println!("Data:{}", String::from_utf8_lossy(&self.data));
        println!("Previous Block Hash:{}", utils::hex_string(&self.pre_block_hash));
        println!("Hash:{}",utils::hex_string(&self.hash));
    }
}

//change byte code to hex string
