import tempfile
import unittest
from pathlib import Path

from serve_loss import make_app, snapshot


class LossPageTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.smoke = Path(tmp.name) / 'smoke' / 'loss_log.jsonl'
        self.train = Path(tmp.name) / 'train' / 'loss_log.jsonl'
        self.app = make_app({'/': self.smoke, '/train': self.train})

    def request(self, route='/', method='GET'):
        headers = []
        body = b''.join(self.app({'PATH_INFO': route, 'REQUEST_METHOD': method},
                                 lambda *args: headers.extend(args))).decode()
        return headers[0], dict(headers[1]), body

    def test_wait_without_creating_training_dir(self):
        status, headers, body = self.request()
        self.assertEqual(status, '200 OK')
        self.assertEqual(headers['Cache-Control'], 'no-store')
        self.assertIn('等待首条日志', body)
        self.assertFalse(self.smoke.parent.exists())

    def test_sparse_metrics_and_refresh_after_partial_line(self):
        self.smoke.parent.mkdir()
        self.smoke.write_bytes(b'{"step":0,"action_loss_avg":0.12}\n{"step":0,"dynamics_loss_avg":0.34}\n'
                               b'{"step":1,"loss":0.46,"learning_rate":1e-5}\n{"step":2,"loss":')
        data, _, note = snapshot(self.smoke)
        self.assertEqual(note, '')
        self.assertEqual(data['action_loss_avg'], {0: .12})
        self.assertEqual(data['dynamics_loss_avg'], {0: .34})
        self.assertEqual(data['loss'], {1: .46})
        self.assertEqual(self.request()[2].count('<svg '), 4)
        with self.smoke.open('ab') as out:
            out.write(b'0}\n')
        self.assertEqual(snapshot(self.smoke)[0]['loss'][2], 0)
        self.assertIn('最新记录：step 2', self.request()[2])
        self.assertIn('等待首条日志', self.request('/train')[2])

    def test_nonfinite_and_bad_lines(self):
        self.train.parent.mkdir()
        self.train.write_bytes(b'invalid\n{"step":1,"loss":1}\n{"step":2,"loss":NaN}\n'
                               b'{"step":3,"loss":1,"action_loss_avg":Infinity}\n')
        data, _, note = snapshot(self.train)
        self.assertEqual(data['loss'], {1: 1, 2: None, 3: 1})
        self.assertIsNone(data['action_loss_avg'][3])
        self.assertIn('2 个', note)
        self.assertIn('1 条无效记录', note)
        self.assertIn('非有限', self.request('/train')[2])

    def test_only_named_read_only_routes(self):
        self.assertEqual(self.request('/../../etc/passwd')[0], '404 Not Found')
        self.assertEqual(self.request(method='POST')[0], '405 Method Not Allowed')
        self.assertEqual(self.request(method='HEAD')[2], '')


if __name__ == '__main__':
    unittest.main()
