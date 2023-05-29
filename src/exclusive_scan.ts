import addBlockSums from "./exclusive_scan_add_block_sums.wgsl";
import prefixSum from "./exclusive_scan_prefix_sum.wgsl";
import prefixSumBlocks from "./exclusive_scan_prefix_sum_blocks.wgsl";
import {alignTo, compileShader} from "./util";

// Note: This also means the min size we can scan is 128 elements
const SCAN_BLOCK_SIZE = 128;

// Serial scan for validation
export function serialExclusiveScan(array: Uint32Array, output: Uint32Array)
{
    output[0] = 0;
    for (let i = 1; i < array.length; ++i) {
        output[i] = array[i - 1] + output[i - 1];
    }
    return output[array.length - 1] + array[array.length - 1];
}

export class ExclusiveScan
{
    #device: GPUDevice;

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

    private constructor(device: GPUDevice)
    {
        this.#device = device;
    }

    static async create(device: GPUDevice)
    {
        let self = new ExclusiveScan(device);

        let scanAddBGLayout = device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {type: "storage", hasDynamicOffset: true}
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

        let scanBlockBGLayout = device.createBindGroupLayout({
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
                bindGroupLayouts: [scanAddBGLayout],
            }),
            compute: {
                module: await compileShader(device, prefixSum, "ExclusiveScan::prefixSum"),
                entryPoint: "main",
                constants: {"0": SCAN_BLOCK_SIZE}
            }
        });

        self.#scanBlockResultsPipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({
                bindGroupLayouts: [scanBlockBGLayout],
            }),
            compute: {
                module: await compileShader(
                    device, prefixSumBlocks, "ExclusiveScan::prefixSumBlocks"),
                entryPoint: "main",
                constants: {"0": SCAN_BLOCK_SIZE}
            }
        });

        self.#addBlockSumsPipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({
                bindGroupLayouts: [scanAddBGLayout],
            }),
            compute: {
                module:
                    await compileShader(device, addBlockSums, "ExclusiveScan::addBlockSums"),
                entryPoint: "main",
                constants: {"0": SCAN_BLOCK_SIZE}
            }
        });
        return self;
    }

    getAlignedSize(size: number)
    {
        return alignTo(size, SCAN_BLOCK_SIZE);
    }

    async scan(buffer: GPUBuffer, size: number)
    {
        const bufferTotalSize = buffer.size / 4;
        if (bufferTotalSize != this.getAlignedSize(bufferTotalSize)) {
            throw Error(`Error: GPU input buffer size (${bufferTotalSize}) must be aligned to ExclusiveScan::getAlignedSize, expected ${this.getAlignedSize(bufferTotalSize)}`)
        }

        let readbackBuf = this.#device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        let blockSumBuf = this.#device.createBuffer({
            size: SCAN_BLOCK_SIZE * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        let carryBuf = this.#device.createBuffer({
            size: 8,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        })
        let carryIntermediateBuf = this.#device.createBuffer({
            size: 4,
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

        let scanBlocksBG = this.#device.createBindGroup({
            layout: this.#scanBlocksPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: buffer,
                        size: this.#maxScanSize * 4,
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: blockSumBuf,
                    },
                },
            ],
        });

        let scanRemainderBlocksBG = this.#device.createBindGroup({
            layout: this.#scanBlocksPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: buffer,
                        size: (bufferTotalSize % this.#maxScanSize) * 4,
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: blockSumBuf,
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
            // range checking so that we don't have to touch regions of the buffer you don't
            // tell us to
            commandEncoder.clearBuffer(buffer, size * 4, 4);
        }

        // Record the scan commands
        for (var i = 0; i < numChunks; ++i) {
            let currentScanBlocksBG = i + 1 < numChunks ? scanBlocksBG : scanRemainderBlocksBG;

            let nWorkGroups = Math.min(
                (bufferTotalSize - i * this.#maxScanSize) / SCAN_BLOCK_SIZE, SCAN_BLOCK_SIZE);

            // Clear the previous block sums
            commandEncoder.clearBuffer(blockSumBuf);

            var computePass = commandEncoder.beginComputePass();

            computePass.setPipeline(this.#scanBlocksPipeline);
            computePass.setBindGroup(0, currentScanBlocksBG, [i * this.#maxScanSize * 4]);
            computePass.dispatchWorkgroups(nWorkGroups, 1, 1);

            computePass.setPipeline(this.#scanBlockResultsPipeline);
            computePass.setBindGroup(0, scanBlockResultsBG);
            computePass.dispatchWorkgroups(1, 1, 1);

            computePass.setPipeline(this.#addBlockSumsPipeline);
            computePass.setBindGroup(0, currentScanBlocksBG, [i * this.#maxScanSize * 4]);
            computePass.dispatchWorkgroups(nWorkGroups, 1, 1);

            computePass.end();

            // Update the carry in value for the next chunk, copy carry out to carry in
            commandEncoder.copyBufferToBuffer(carryBuf, 4, carryIntermediateBuf, 0, 4);
            commandEncoder.copyBufferToBuffer(carryIntermediateBuf, 0, carryBuf, 0, 4);
        }

        // Copy the final scan result back to the readback buffer
        if (size < bufferTotalSize) {
            commandEncoder.copyBufferToBuffer(buffer, size * 4, readbackBuf, 0, 4);
        } else {
            commandEncoder.copyBufferToBuffer(carryBuf, 4, readbackBuf, 0, 4);
        }

        this.#device.queue.submit([commandEncoder.finish()]);
        await this.#device.queue.onSubmittedWorkDone();

        await readbackBuf.mapAsync(GPUMapMode.READ);
        var mapping = new Uint32Array(readbackBuf.getMappedRange());
        var sum = mapping[0];
        readbackBuf.unmap();

        return sum;
    }
};
