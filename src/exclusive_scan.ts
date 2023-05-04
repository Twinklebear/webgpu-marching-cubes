import {alignTo, compileShader} from "./volume";

import addBlockSums from "./exclusive_scan_add_block_sums.wgsl";
import prefixSum from "./exclusive_scan_prefix_sum.wgsl";
import prefixSumBlocks from "./exclusive_scan_prefix_sum_blocks.wgsl";

const SCAN_BLOCK_SIZE = 512;

export class ExclusiveScan {
    #device: GPUDevice;

    // Note: really should be set/read/generated in the scanner shader code
    // Here just have it hard-coded
    // We scan chunks of 512 elements, wgsize = ScanBlockSize / 2
    readonly #workGroupSize = SCAN_BLOCK_SIZE / 2;

    // The max # of elements that can be scanned without carry in/out
    readonly #maxScanSize = SCAN_BLOCK_SIZE * SCAN_BLOCK_SIZE;

    // Pipeline for scanning the individual blocks of ScanBlockSize elements 
    #scanBlocksPipeline: GPUComputePipeline;

    // Pipeline for scanning the block scan results which will then be added back to
    // the individual block scan results
    #scanBlockResultsPipeline: GPUComputePipeline;

    // Pipeline that adds the block scan results back to each individual block so
    // that its scan result is globally correct based on the elements preceeding the block
    #addBlockSumsPipeline: GPUComputePipeline;

    private constructor(device: GPUDevice) {
        this.#device = device;
    }

    static async create(device: GPUDevice) {
        let self = new ExclusiveScan(device);

        // TODO: maybe use a dynamic offset again? I think I was using that before
        // they got disabled temporarily
        let bindGroupLayout = device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage",
                    }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage",
                    }
                },
            ],
        });

        self.#scanBlocksPipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({
                bindGroupLayouts: [bindGroupLayout],
            }),
            compute: {
                module: await compileShader(device, prefixSum, "ExclusiveScan::prefixSum"),
                entryPoint: "main",
                constants: {
                    "0": SCAN_BLOCK_SIZE
                }
            }
        });

        self.#scanBlockResultsPipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({
                bindGroupLayouts: [bindGroupLayout],
            }),
            compute: {
                module: await compileShader(device, prefixSumBlocks, "ExclusiveScan::prefixSumBlocks"),
                entryPoint: "main",
                constants: {
                    "0": SCAN_BLOCK_SIZE
                }
            }
        });

        self.#addBlockSumsPipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({
                bindGroupLayouts: [bindGroupLayout],
            }),
            compute: {
                module: await compileShader(device, addBlockSums, "ExclusiveScan::addBlockSums"),
                entryPoint: "main",
                constants: {
                    "0": SCAN_BLOCK_SIZE
                }
            }
        });
        return self;
    }

    getAlignedSize(size: number) {
        return alignTo(size, SCAN_BLOCK_SIZE);
    }

    async scan(buffer: GPUBuffer, size: number) {
        const bufferTotalSize = buffer.size;
        // TODO: merge the prep gpu input and scan to make a lot of the buffers
        // into just small temporaries
        if (bufferTotalSize != this.getAlignedSize(size)) {
            throw Error("Error: GPU input buffer size must be aligned to ExclusiveScan::getAlignedSize")
        }

        let readbackBuffer = this.#device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        // TODO: run a clear on this buffer 
        let blockSumBuf = this.#device.createBuffer({
            size: SCAN_BLOCK_SIZE * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        // TODO: run a clear on carry in/out buffer
        // TODO: Also jsut use the readbackBuffer as the extra copy space to avoid aliasing
        let carryBuf = this.#device.createBuffer({
            size: 8,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        })

        let scanBlockResultsBG = this.#device.createBindGroup({
            layout: this.#scanBlockResultsPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: blockSumBuf,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: carryBuf,
                    },
                },
            ],
        });

        const numChunks = Math.ceil(size / this.#maxScanSize);

        var commandEncoder = this.#device.createCommandEncoder();
        commandEncoder.clearBuffer(blockSumBuf);
        commandEncoder.clearBuffer(carryBuf);
        // If the size being scanned is less than the buffer size, clear the end of it
        // so we don't pull down invalid values
        if (size < bufferTotalSize) {
            // TODO: Later the scan should support not reading these values by doing proper
            // range checking so that we don't have to touch regions of the buffer you don't tell us to
            commandEncoder.clearBuffer(buffer, size * 4, (bufferTotalSize - size) * 4);
        }

        // Record the scan commands
        for (var i = 0; i < numChunks; ++i) {

        }
    }
};

/*

var ExclusiveScanner = function (scanPipeline, gpuBuffer, alignedSize) {
    this.scanPipeline = scanPipeline;
    this.inputSize = alignedSize;
    this.inputBuf = gpuBuffer;

    // TODO: Make this a temporary
    this.readbackBuf = scanPipeline.device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // TODO: Make this a temporary
    // Block sum buffer
    var blockSumBuf = scanPipeline.device.createBuffer({
        size: ScanBlockSize * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Uint32Array(blockSumBuf.getMappedRange()).fill(0);
    blockSumBuf.unmap();
    this.blockSumBuf = blockSumBuf;

    // TODO: Make this a temporary
    var carryBuf = scanPipeline.device.createBuffer({
        size: 8,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Uint32Array(carryBuf.getMappedRange()).fill(0);
    carryBuf.unmap();
    this.carryBuf = carryBuf;

    // TODO: Make this a temporary
    // Can't copy from a buffer to itself so we need an intermediate to move the carry
    this.carryIntermediateBuf = scanPipeline.device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    // TODO: Make this a temporary
    this.scanBlockResultsBindGroup = scanPipeline.device.createBindGroup({
        layout: this.scanPipeline.scanBlockResultsLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: blockSumBuf,
                },
            },
            {
                binding: 1,
                resource: {
                    buffer: carryBuf,
                },
            },
        ],
    });
};

ExclusiveScanner.prototype.scan = async function (dataSize) {
    // If the data size we're scanning within the larger input array has changed,
    // we just need to re-record the scan commands
    var numChunks = Math.ceil(dataSize / this.scanPipeline.maxScanSize);
    this.offsets = new Uint32Array(numChunks);
    for (var i = 0; i < numChunks; ++i) {
        this.offsets.set([i * this.scanPipeline.maxScanSize * 4], i);
    }

    // Scan through the data in chunks, updating carry in/out at the end to carry
    // over the results of the previous chunks
    var commandEncoder = this.scanPipeline.device.createCommandEncoder();

    // Clear the carry buffer and the readback sum entry if it's not scan size aligned
    commandEncoder.copyBufferToBuffer(this.scanPipeline.clearBuf, 0, this.carryBuf, 0, 8);
    for (var i = 0; i < numChunks; ++i) {
        var nWorkGroups =
            Math.min((this.inputSize - i * this.scanPipeline.maxScanSize) / ScanBlockSize,
                ScanBlockSize);

        var scanBlockBG = null;
        if (nWorkGroups === ScanBlockSize) {
            scanBlockBG = this.scanPipeline.device.createBindGroup({
                layout: this.scanPipeline.scanBlocksLayout,
                entries: [
                    {
                        binding: 0,
                        resource: {
                            buffer: this.inputBuf,
                            size: Math.min(this.scanPipeline.maxScanSize, this.inputSize) * 4,
                            offset: this.offsets[i],
                        },
                    },
                    {
                        binding: 1,
                        resource: {
                            buffer: this.blockSumBuf,
                        },
                    },
                ],
            });
        } else {
            // Bind groups for processing the remainder if the aligned size isn't
            // an even multiple of the max scan size
            scanBlockBG = this.scanPipeline.device.createBindGroup({
                layout: this.scanPipeline.scanBlocksLayout,
                entries: [
                    {
                        binding: 0,
                        resource: {
                            buffer: this.inputBuf,
                            size: (this.inputSize % this.scanPipeline.maxScanSize) * 4,
                            offset: this.offsets[i],
                        },
                    },
                    {
                        binding: 1,
                        resource: {
                            buffer: this.blockSumBuf,
                        },
                    },
                ],
            });
        }

        // Clear the previous block sums
        commandEncoder.copyBufferToBuffer(
            this.scanPipeline.clearBuf, 0, this.blockSumBuf, 0, ScanBlockSize * 4);

        var computePass = commandEncoder.beginComputePass();

        computePass.setPipeline(this.scanPipeline.scanBlocksPipeline);
        computePass.setBindGroup(0, scanBlockBG);
        computePass.dispatchWorkgroups(nWorkGroups, 1, 1);

        computePass.setPipeline(this.scanPipeline.scanBlockResultsPipeline);
        computePass.setBindGroup(0, this.scanBlockResultsBindGroup);
        computePass.dispatchWorkgroups(1, 1, 1);

        computePass.setPipeline(this.scanPipeline.addBlockSumsPipeline);
        computePass.setBindGroup(0, scanBlockBG);
        computePass.dispatchWorkgroups(nWorkGroups, 1, 1);

        computePass.end();

        // Update the carry in value for the next chunk, copy carry out to carry in
        commandEncoder.copyBufferToBuffer(this.carryBuf, 4, this.carryIntermediateBuf, 0, 4);
        commandEncoder.copyBufferToBuffer(this.carryIntermediateBuf, 0, this.carryBuf, 0, 4);
    }
    var commandBuffer = commandEncoder.finish();

    // We need to clear a different element in the input buf for the last item if the data size
    // shrinks
    if (dataSize < this.inputSize) {
        var commandEncoder = this.scanPipeline.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(
            this.scanPipeline.clearBuf, 0, this.inputBuf, dataSize * 4, 4);
        this.scanPipeline.device.queue.submit([commandEncoder.finish()]);
    }

    this.scanPipeline.device.queue.submit([commandBuffer]);

    // Readback the the last element to return the total sum as well
    var commandEncoder = this.scanPipeline.device.createCommandEncoder();
    if (dataSize < this.inputSize) {
        commandEncoder.copyBufferToBuffer(this.inputBuf, dataSize * 4, this.readbackBuf, 0, 4);
    } else {
        commandEncoder.copyBufferToBuffer(this.carryBuf, 4, this.readbackBuf, 0, 4);
    }
    this.scanPipeline.device.queue.submit([commandEncoder.finish()]);

    await this.readbackBuf.mapAsync(GPUMapMode.READ);
    var mapping = new Uint32Array(this.readbackBuf.getMappedRange());
    var sum = mapping[0];
    this.readbackBuf.unmap();

    return sum;
};

*/
