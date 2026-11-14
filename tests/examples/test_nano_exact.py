import base64
import importlib.util
import json
import os
import tempfile
import time
import unittest

_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "examples",
    "guides",
    "x402_examples",
    "nano_exact",
    "nano_exact.py",
)
_spec = importlib.util.spec_from_file_location("nano_exact", _PATH)
nx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nx)

BURN = "nano_1111111111111111111111111111111111111111111111111111hifc8npp"
PRICE = nx.RAW_PER_XNO // 20
HASH_A = "A" * 64
HASH_B = "b" * 64


class FakeNode:
    def __init__(self):
        self.blocks = {}

    def read_block(self, block_hash):
        if block_hash not in self.blocks:
            raise ValueError("block not found")
        return self.blocks[block_hash]


def send_block(order_id, pay_to=BURN, amount=PRICE, **over):
    info = {
        "subtype": "send",
        "confirmed": "true",
        "amount": str(amount),
        "local_timestamp": str(int(time.time())),
        "contents": {
            "link_as_account": pay_to,
            "representative": nx.commitment_rep_account(
                order_id, pay_to, amount
            ),
        },
    }
    info.update(over)
    return info


def header(block_hash, order_id, network=nx.NETWORK):
    payment = {
        "x402Version": 1,
        "scheme": "exact",
        "network": network,
        "payload": {"blockHash": block_hash, "orderId": order_id},
    }
    return base64.b64encode(json.dumps(payment).encode()).decode()


class NanoExactTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.node = FakeNode()
        self.seller = self.new_seller()

    def tearDown(self):
        self.tmp.cleanup()

    def new_seller(self, **kw):
        return nx.NanoExactSeller(
            BURN, PRICE, self.node, self.tmp.name, **kw
        )

    def paid_order(self, resource="/research?query=q", block=HASH_A):
        entry = self.seller.offer(resource)["accepts"][0]
        order_id = entry["extra"]["orderId"]
        self.node.blocks[block.upper()] = send_block(order_id)
        return order_id

    def test_binding_vector_from_the_standard(self):
        self.assertEqual(
            nx.commitment_rep_account(
                "order-0001", BURN, 50000000000000000000000000000
            ),
            "nano_33q9dassq66es6mcay83n5kypqgpzjikr9bkc9ope9xch1borfphgehj4ozj",
        )

    def test_account_checksum(self):
        self.assertTrue(nx.is_valid_account(BURN))
        self.assertFalse(nx.is_valid_account(BURN[:-1] + "q"))
        with self.assertRaises(ValueError):
            nx.NanoExactSeller(
                BURN[:-1] + "q", PRICE, self.node, self.tmp.name
            )

    def test_price_must_be_integer_raw(self):
        for bad in (0.05, True, -1, "1"):
            with self.assertRaises(TypeError):
                nx.NanoExactSeller(
                    BURN, bad, self.node, self.tmp.name
                )

    def test_offer_shape(self):
        entry = self.seller.offer("/research?query=q")["accepts"][0]
        self.assertEqual(entry["scheme"], "exact")
        self.assertEqual(entry["network"], "nano:mainnet")
        self.assertEqual(entry["maxAmountRequired"], str(PRICE))
        self.assertEqual(entry["payTo"], BURN)
        order_id = entry["extra"]["orderId"]
        self.assertGreaterEqual(len(order_id), 32)
        self.assertEqual(
            entry["extra"]["representative"],
            nx.commitment_rep_account(order_id, BURN, PRICE),
        )

    def test_paid_once_then_replay_refused(self):
        order_id = self.paid_order()
        h = header(HASH_A, order_id)
        self.assertEqual(
            self.seller.verify(h, "/research?query=q"), nx.MATCHED
        )
        self.assertEqual(
            self.seller.verify(h, "/research?query=q"),
            nx.ALREADY_CLAIMED,
        )

    def test_replay_refused_after_restart(self):
        order_id = self.paid_order()
        h = header(HASH_A, order_id)
        self.assertEqual(
            self.seller.verify(h, "/research?query=q"), nx.MATCHED
        )
        restarted = self.new_seller()
        self.assertEqual(
            restarted.verify(h, "/research?query=q"),
            nx.ALREADY_CLAIMED,
        )

    def test_order_survives_restart(self):
        order_id = self.paid_order()
        restarted = self.new_seller()
        self.assertEqual(
            restarted.verify(
                header(HASH_A, order_id), "/research?query=q"
            ),
            nx.MATCHED,
        )

    def test_block_bound_to_another_order_is_refused(self):
        paid = self.paid_order()
        other = self.seller.offer("/research?query=q")["accepts"][0]
        other_id = other["extra"]["orderId"]
        self.assertNotEqual(paid, other_id)
        self.assertEqual(
            self.seller.verify(
                header(HASH_A, other_id), "/research?query=q"
            ),
            nx.NOT_BOUND,
        )

    def test_order_used_for_another_request_is_refused(self):
        order_id = self.paid_order()
        self.assertEqual(
            self.seller.verify(
                header(HASH_A, order_id), "/research?query=other"
            ),
            nx.WRONG_RESOURCE,
        )

    def test_second_block_for_a_paid_order_is_refused(self):
        order_id = self.paid_order()
        self.node.blocks[HASH_B.upper()] = send_block(order_id)
        resource = "/research?query=q"
        self.assertEqual(
            self.seller.verify(header(HASH_A, order_id), resource),
            nx.MATCHED,
        )
        self.assertEqual(
            self.seller.verify(header(HASH_B, order_id), resource),
            nx.ALREADY_CLAIMED,
        )

    def test_block_checks(self):
        entry = self.seller.offer("/r")["accepts"][0]
        oid = entry["extra"]["orderId"]
        other = nx.public_key_to_account(b"\x01" * 32)
        cases = {
            nx.WRONG_AMOUNT: send_block(oid, amount=PRICE - 1),
            nx.WRONG_PAY_TO: send_block(oid, pay_to=other),
            nx.NOT_CONFIRMED: send_block(oid, confirmed="false"),
            nx.NOT_A_SEND: send_block(oid, subtype="receive"),
            nx.LATE: send_block(
                oid,
                local_timestamp=str(entry["extra"]["expiresAt"] + 1),
            ),
            nx.INCONCLUSIVE: send_block(oid, local_timestamp="0"),
        }
        for expected, info in cases.items():
            self.node.blocks[HASH_A] = info
            self.assertEqual(
                self.seller.verify(header(HASH_A, oid), "/r"),
                expected,
                expected,
            )

    def test_bad_headers(self):
        order_id = self.paid_order(resource="/r")
        self.assertEqual(
            self.seller.verify("not base64!", "/r"),
            nx.INVALID_PAYMENT,
        )
        self.assertEqual(
            self.seller.verify(
                header(HASH_A, order_id, network="other"), "/r"
            ),
            nx.INVALID_PAYMENT,
        )
        self.assertEqual(
            self.seller.verify(header("12", order_id), "/r"),
            nx.INVALID_PAYMENT,
        )
        self.assertEqual(
            self.seller.verify(header(HASH_A, "nope"), "/r"),
            nx.UNKNOWN_ORDER,
        )
        self.assertEqual(
            self.seller.verify(header(HASH_B, order_id), "/r"),
            nx.INCONCLUSIVE,
        )

    def test_node_url_rules(self):
        nx.NanoNode("https://node.example")
        nx.NanoNode("http://127.0.0.1:7076")
        nx.NanoNode("http://localhost:7076")
        for bad in ("http://node.example", "ftp://node.example"):
            with self.assertRaises(ValueError):
                nx.NanoNode(bad)


if __name__ == "__main__":
    unittest.main()
