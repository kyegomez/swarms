use std::{env, process::Command};

use crate::blockchain::Blockchain;

pub struct CLI {
    bc: Blockchain,
}

impl CLI {
    pub fn new(bc :Blockchain) -> Self {
        CLI {bc}
    }

    pub fn run(&mut self) {
        let args: Vec<String> = env::args().collect();
        if args.len() < 2 {
            self.print_usage();
            return;
        }

        let command = &args[1];
        match command.as_str() {
            "addblock" => {
                if args.len() !=3 {
                    println!("Usage :addblock <DATA>");
                } else {
                    self.add_block(&args[2]);
                }
            },
            "printchain" => {
                self.bc.print_block();
            },
            _ => {
                self.print_usage();
            },
        }
    }

    fn print_usage(&self) {
        println!("Usage:");
        println!("  addblock <DATA> -add a block to the blcokchain");
        println!("  printchain - print all the blocks of the blockchain");
    }

    fn add_block(&mut self, data: &str) {
        self.bc.add_block(data);
        println!("Sucess!");
    }
}

