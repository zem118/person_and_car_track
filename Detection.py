
class Detection:
    def __init__(self, cls_name, ltrb, track_id):
        self.cls_name = cls_name
        self.track_id = track_id

        self.ltrb = ltrb
        self.pre_ltrb = None
        self.track_list = []
        self.direction = None

        self.update_track_list()

    def update(self, ltrb):
        self.pre_ltrb = self.ltrb
        self.ltrb = ltrb

        self.update_track_list()
        self.update_direction()

    def update_track_list(self):
        l, t, r, b = self.ltrb
        center_x = (l + r) // 2

        self.track_list.append((center_x, b))
        if len(self.track_list) > 20:
            del self.track_list[0]

    def get_track_list(self):
        return self.track_list

    def update_direction(self):
        if len(self.track_list) >= 20:
            x_delta = self.track_list[-1][0] - self.track_list[0][0]

            if x_delta > 25:
                self.direction = 'right'
            elif x_delta < -25:
                self.direction = 'left'
            else:
                self.direction = ''
