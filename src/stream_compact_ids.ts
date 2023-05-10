import {PushConstants} from "./push_constant_builder";
import {compileShader} from "./volume";

import streamCompactIDs from "./stream_compact_ids.wgsl";

// Serial version for validation
export function serialStreamCompactIDs(isActiveBuffer: Uint32Array, offsetBuffer: Uint32Array, idOutputBuffer: Uint32Array) {
    for (let i = 0; i < isActiveBuffer.length; ++i) {
        if (isActiveBuffer[i] != 0) {
            idOutputBuffer[offsetBuffer[i]] = i;
        }
    }
}

export class StreamCompactIDs {
    #device: GPUDevice;

    // Should be at least 64 so that we process elements
    // in 256b blocks with each WG. This will ensure that our
    // dynamic offsets meet the 256b alignment requirement
    readonly WORKGROUP_SIZE: number = 64;

    readonly #maxDispatchSize: number;

    #computePipeline: GPUComputePipeline;

    private constructor(device: GPUDevice) {
        this.#device = device;

        this.#maxDispatchSize = device.limits.maxComputeWorkgroupsPerDimension;
    }

    static async create(device: GPUDevice) {
        let self = new StreamCompactIDs(device);

        let paramsBGLayout = device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage",
                        hasDynamicOffset: true,
                    }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage",
                        hasDynamicOffset: true,
                    }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage",
                    }
                },
            ],
        });

        let pushConstantsBGLayout = device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "uniform",
                        hasDynamicOffset: true
                    }
                },
            ]
        });

        self.#computePipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({
                bindGroupLayouts: [paramsBGLayout, pushConstantsBGLayout]
            }),
            compute: {
                module: await compileShader(device, streamCompactIDs, "StreamCompactIDs"),
                entryPoint: "main",
                constants: {
                    "0": self.WORKGROUP_SIZE
                }
            }
        });

        return self;
    }

    async compactActiveIDs(isActiveBuffer: GPUBuffer, offsetBuffer: GPUBuffer, idOutputBuffer: GPUBuffer,
        size: number) {

        // Build the push constants
        let pushConstantsArg = new Uint32Array([size]);
        let pushConstants = new PushConstants(this.#device, Math.ceil(size / this.WORKGROUP_SIZE),
            pushConstantsArg.buffer);

        let pushConstantsBG = this.#device.createBindGroup({
            layout: this.#computePipeline.getBindGroupLayout(1),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: pushConstants.pushConstantsBuffer,
                        size: 12,
                    }
                }
            ]
        });

        // # of elements we can compact in a single dispatch.
        const elementsPerDispatch = this.#maxDispatchSize * this.WORKGROUP_SIZE;

        // Ensure we won't break the dynamic offset alignment rules
        if (pushConstants.numDispatches() > 1 && (elementsPerDispatch * 4) % 256 != 0) {
            throw Error("StreamCompactIDs: Buffer dynamic offsets will not be 256b aligned! Set WORKGROUP_SIZE = 64");
        }

        // With dynamic offsets the size/offset validity checking means we still need to
        // create a separate bind group for the remainder elements that don't evenly fall into
        // a full size dispatch
        let paramsBG = this.#device.createBindGroup({
            layout: this.#computePipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: isActiveBuffer,
                        size: Math.min(size, elementsPerDispatch) * 4,
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: offsetBuffer,
                        size: Math.min(size, elementsPerDispatch) * 4,
                    }
                },
                {
                    binding: 2,
                    resource: {
                        buffer: idOutputBuffer,
                    }
                }
            ]
        });


        // Make a remainder elements bindgroup if we have some remainder to make sure
        // we don't bind out of bounds regions of the buffer. If there's no remiander we
        // just set remainderParamsBG to paramsBG so that on our last dispatch we can just
        // always bindg remainderParamsBG
        let remainderParamsBG = paramsBG;
        const remainderElements = size % elementsPerDispatch;
        if (remainderElements != 0) {
            // Note: We don't set the offset here, as that will still be handled by the
            // dynamic offsets. We just need to set the right size, so that
            // dynamic offset + binding size is >= buffer size
            remainderParamsBG = this.#device.createBindGroup({
                layout: this.#computePipeline.getBindGroupLayout(0),
                entries: [
                    {
                        binding: 0,
                        resource: {
                            buffer: isActiveBuffer,
                            size: remainderElements * 4,
                        }
                    },
                    {
                        binding: 1,
                        resource: {
                            buffer: offsetBuffer,
                            size: remainderElements * 4,
                        }
                    },
                    {
                        binding: 2,
                        resource: {
                            buffer: idOutputBuffer,
                        }
                    }
                ]
            });
        }


        let commandEncoder = this.#device.createCommandEncoder();
        let pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.#computePipeline);
        for (let i = 0; i < pushConstants.numDispatches(); ++i) {
            let dispatchParamsBG = paramsBG;
            if (i + 1 == pushConstants.numDispatches()) {
                dispatchParamsBG = remainderParamsBG;
            }
            pass.setBindGroup(0, dispatchParamsBG, [i * elementsPerDispatch * 4, i * elementsPerDispatch * 4]);
            pass.setBindGroup(1, pushConstantsBG, [i * pushConstants.stride]);
            pass.dispatchWorkgroups(pushConstants.dispatchSize(i), 1, 1);
        }
        pass.end();
        this.#device.queue.submit([commandEncoder.finish()]);
        await this.#device.queue.onSubmittedWorkDone();
    }
}

