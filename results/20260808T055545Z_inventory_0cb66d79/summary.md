# Jaguar Inventory Summary

- Run: `20260808T055545Z_inventory_0cb66d79`
- Board: `BirmanPlus-KRK2e` `RevB`
- BIOS: `RIJ0071C` (11/10/2025 20:15:08)
- CPU: Ryzen Embedded V4A46X, 6 cores / 12 threads, amd-pstate-epp
- GPU: `0x1002:0x1902` via `amdgpu`
- NPU: `0x1022:0x17f0 rev 0x20` via `amdxdna`
- NPU generation: KRK/XDNA2
- DRM connectors: 10 exposed, 1 connected
- IOMMU groups: 21
- Active network: ax88179_178a (USB 2.0 480 Mb/s bus with 1 GbE PHY)

## Blockers

- **display**: only 1 physical connector is connected; 4-6 sinks are required
- **sfp_plus**: no PCI network-class function or native network interface is active
- **npu_telemetry**: xrt-smi is not installed; amdxdna presence alone does not prove runtime compatibility
- **oi_quantization**: representative segmentation calibration/evaluation data and preprocessing are not supplied

## Missing Tools

`xrt-smi`
