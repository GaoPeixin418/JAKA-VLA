import unittest
from jaka_vla.contact_acceptance import lab_forbidden_contact


class LabContactAcceptance(unittest.TestCase):
    def test_only_intended_robot_contacts_are_accepted(self):
        for pair in [('scene:box','scene:table'),('left:finger_third','scene:box'),('left:Link_0','scene:table')]:
            self.assertFalse(lab_forbidden_contact(pair), pair)
        for pair in [('left:finger_left','scene:distractor_1'),('left:tool','scene:box'),
                     ('left:Link_1','scene:table'),('right:finger_left','scene:box'),
                     ('left:finger_left','right:Link_2'),('left:camera','left:finger_right')]:
            self.assertTrue(lab_forbidden_contact(pair), pair)


if __name__ == '__main__':
    unittest.main()
