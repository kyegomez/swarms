use crate::block::{Block, self};

pub struct Blockchain {
    pub blocks: Vec<Block>,
}

impl Blockchain {
    pub(crate) fn new() -> Blockchain {
        let genesis_block = Block::new_genesis_block();
        Blockchain { blocks: vec![genesis_block] }
    }

    pub fn add_block(&mut self, data: &str) {
        let pre_block :&Block = self.blocks.last().unwrap().clone();
        let pre_block_hash = pre_block.hash.clone();
        let new_block = Block::new_block_chain(data, pre_block_hash);

        self.blocks.push(new_block)
    }

    pub fn print_block(&self) {
        for block in self.blocks.iter() {
        block.print_content();
        println!("----------------");
      }
    }
}