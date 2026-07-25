from dataclasses import dataclass, field

@dataclass
class TrackState:
    last_seen: int = -1
    confirmed_frames: int = 0
    inside: bool = False
    inside_streak: int = 0
    outside_streak: int = 0

@dataclass
class EventEngine:
    inside_req: int = 2
    outside_req: int = 2
    min_confirm_frames: int = 3
    max_missed_inside: int = 20
    states: dict = field(default_factory=dict)

    def update_track(self, track_id, frame_idx, inside_now):
        st = self.states.setdefault(track_id, TrackState())
        st.last_seen = frame_idx
        st.confirmed_frames += 1

        if inside_now:
            st.inside_streak += 1
            st.outside_streak = 0
        else:
            st.outside_streak += 1
            st.inside_streak = 0

        event = None
        if st.confirmed_frames >= self.min_confirm_frames:
            if (not st.inside) and st.inside_streak >= self.inside_req:
                st.inside = True
                event = "ENTER"
            elif st.inside and st.outside_streak >= self.outside_req:
                st.inside = False
                event = "EXIT"

        return event

    def finalize_missed(self, frame_idx):
        
        forced_exits = []
        for tid, st in self.states.items():
            if st.inside and (frame_idx - st.last_seen > self.max_missed_inside):
                st.inside = False
                forced_exits.append(tid)
        return forced_exits

    def current_occupancy(self):
        return sum(1 for s in self.states.values() if s.inside)
