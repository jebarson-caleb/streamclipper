"""Keep trying to create the free Oracle A1 server until capacity frees up.

Usage (uses ~/.oci/config):
    python scripts/oracle_grab_a1.py --dry-run   # only look up image/subnet/AD, create nothing
    python scripts/oracle_grab_a1.py             # retry every --interval seconds until it succeeds
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import oci

p = argparse.ArgumentParser()
p.add_argument("--name", default="streamclipper")
p.add_argument("--ocpus", type=float, default=2)
p.add_argument("--memory", type=float, default=12)
p.add_argument("--boot-gb", type=int, default=100)
p.add_argument("--ssh-key", default=str(Path.home() / ".ssh" / "oracle_streamclipper.pub"))
p.add_argument("--interval", type=int, default=120, help="seconds between attempts")
p.add_argument("--dry-run", action="store_true")
args = p.parse_args()

SHAPE = "VM.Standard.A1.Flex"
cfg = oci.config.from_file()
tenancy = cfg["tenancy"]  # root compartment
compute = oci.core.ComputeClient(cfg)
network = oci.core.VirtualNetworkClient(cfg)
identity = oci.identity.IdentityClient(cfg)


def log(msg: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}", flush=True)


def existing() -> list:
    return [i for i in compute.list_instances(tenancy).data
            if i.display_name == args.name and i.lifecycle_state not in ("TERMINATED", "TERMINATING")]


ads = [a.name for a in identity.list_availability_domains(tenancy).data]
images = compute.list_images(tenancy, operating_system="Canonical Ubuntu", operating_system_version="24.04",
                             shape=SHAPE, sort_by="TIMECREATED", sort_order="DESC").data
images = [i for i in images if "Minimal" not in i.display_name]
subnets = [s for s in network.list_subnets(tenancy).data
           if s.lifecycle_state == "AVAILABLE" and not s.prohibit_public_ip_on_vnic]
if not images or not subnets:
    raise SystemExit(f"lookup failed: images={len(images)} public subnets={len(subnets)}")
image, subnet = images[0], subnets[0]
log(f"ADs: {ads}")
log(f"image: {image.display_name}")
log(f"subnet: {subnet.display_name}")
log(f"shape: {SHAPE} {args.ocpus:g} OCPU / {args.memory:g} GB, boot {args.boot_gb} GB")
if existing():
    raise SystemExit(f"an instance named {args.name!r} already exists; nothing to do")
if args.dry_run:
    raise SystemExit("dry run: nothing created")

ssh_key = Path(args.ssh_key).read_text().strip()
attempt = 0
while True:
    for ad in ads:
        attempt += 1
        try:
            inst = compute.launch_instance(oci.core.models.LaunchInstanceDetails(
                compartment_id=tenancy,
                availability_domain=ad,
                display_name=args.name,
                shape=SHAPE,
                shape_config=oci.core.models.LaunchInstanceShapeConfigDetails(
                    ocpus=args.ocpus, memory_in_gbs=args.memory),
                source_details=oci.core.models.InstanceSourceViaImageDetails(
                    image_id=image.id, boot_volume_size_in_gbs=args.boot_gb),
                create_vnic_details=oci.core.models.CreateVnicDetails(
                    subnet_id=subnet.id, assign_public_ip=True),
                metadata={"ssh_authorized_keys": ssh_key},
            )).data
            log(f"CREATED {inst.display_name} ({inst.id}) in {ad} after {attempt} attempts")
            raise SystemExit(0)
        except oci.exceptions.ServiceError as e:
            if "capacity" in (e.message or "").lower() or e.status in (429, 500):
                log(f"attempt {attempt}: {ad}: {e.message[:90]}")
            else:
                raise SystemExit(f"stopping, not a capacity error: {e.status} {e.code}: {e.message}")
    time.sleep(args.interval)
