/**
 * ComfyUI GeomPack - Gaussian Splat Preview Widget
 * Interactive 3D Gaussian Splatting viewer using gsplat.js
 */

import { app } from "../../../scripts/app.js";

// Auto-detect extension folder name (handles ComfyUI-GeometryPack or comfyui-geometrypack)
const EXTENSION_FOLDER = (() => {
    const url = import.meta.url;
    const match = url.match(/\/extensions\/([^/]+)\//);
    return match ? match[1] : "ComfyUI_DJBFilmz";
})();

console.log("[DJBFilmz.GaussianPreview] Loading extension...");

const COORDINATE_BASIS_VALUES = new Set(["auto", "worldmirror", "hyworld2_worldgen"]);

function normalizeCoordinateBasis(value) {
    return COORDINATE_BASIS_VALUES.has(value) ? value : "auto";
}

function getCoordinateBasisWidget(node) {
    return node.widgets?.find((w) => w.name === "coordinate_basis");
}

function getConfiguredCoordinateBasis(node, data) {
    const widget = getCoordinateBasisWidget(node);
    const widgetIndex = widget && node.widgets ? node.widgets.indexOf(widget) : -1;
    const widgetValue = widgetIndex >= 0 ? data?.widgets_values?.[widgetIndex] : undefined;
    const propertyValue = data?.properties?.coordinate_basis ?? node.properties?.coordinate_basis;
    if (COORDINATE_BASIS_VALUES.has(widgetValue)) {
        return widgetValue;
    }
    if (COORDINATE_BASIS_VALUES.has(propertyValue)) {
        return propertyValue;
    }
    if (COORDINATE_BASIS_VALUES.has(widget?.value)) {
        return widget.value;
    }
    return "auto";
}

function ensureCoordinateBasisWidget(node, value) {
    const widget = getCoordinateBasisWidget(node);
    if (!widget) {
        return;
    }
    widget.serialize = true;
    widget.options = widget.options || {};
    widget.options.serialize = true;
    node.properties = node.properties || {};
    const nextValue = normalizeCoordinateBasis(value ?? node.properties.coordinate_basis ?? widget.value);
    widget.value = nextValue;
    node.properties.coordinate_basis = nextValue;

    if (!widget._djbfilmzCoordinateBasisPersistent) {
        const originalCallback = widget.callback;
        widget.callback = function (newValue, ...args) {
            const normalized = normalizeCoordinateBasis(newValue);
            widget.value = normalized;
            node.properties = node.properties || {};
            node.properties.coordinate_basis = normalized;
            if (originalCallback) {
                return originalCallback.call(this, normalized, ...args);
            }
        };
        widget._djbfilmzCoordinateBasisPersistent = true;
    }
}

app.registerExtension({
    name: "djbfilmz.gaussianpreview",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "DJBFilmz_HyWorldPreview") {
			console.log("[DJBFilmz.GaussianPreview] Registering Preview Gaussian node");

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                ensureCoordinateBasisWidget(this);

                const node = this;

                // Find the existing widgets automatically created from Python INPUT_TYPES
                const camParamsWidget = this.widgets?.find(w => w.name === "camera_parameters");
                const lockedImageWidget = this.widgets?.find(w => w.name === "locked_image_path");

                // Collapse their size and set hidden state to prevent overlapping/clutter in ComfyUI
                if (camParamsWidget) {
                    camParamsWidget.computeSize = () => [0, -4];
                    camParamsWidget.hidden = true;
                }
                if (lockedImageWidget) {
                    lockedImageWidget.computeSize = () => [0, -4];
                    lockedImageWidget.hidden = true;
                }

                // Create container for viewer + info panel
                const container = document.createElement("div");
                container.style.width = "100%";
                container.style.height = "100%";
                container.style.display = "flex";
                container.style.flexDirection = "column";
                container.style.backgroundColor = "#1a1a1a";
                container.style.overflow = "hidden";
                container.style.marginTop = "8px"; // Spacing below the coordinate_basis dropdown
                container.style.borderRadius = "4px";
                container.style.border = "1px solid #2a2a2a";

                // Create iframe for gsplat.js viewer
                const iframe = document.createElement("iframe");
                iframe.style.width = "100%";
                iframe.style.flex = "1 1 0";
                iframe.style.minHeight = "0";
                iframe.style.border = "none";
                iframe.style.backgroundColor = "#1a1a1a";

                // Point to gsplat.js HTML viewer (with cache buster)
                iframe.src = `/extensions/${EXTENSION_FOLDER}/gaussian_preview/static/viewer_gaussian.html?v=` + Date.now();

                // Create info panel
                const infoPanel = document.createElement("div");
                infoPanel.style.backgroundColor = "#1a1a1a";
                infoPanel.style.borderTop = "1px solid #444";
                infoPanel.style.padding = "6px 12px";
                infoPanel.style.fontSize = "10px";
                infoPanel.style.fontFamily = "monospace";
                infoPanel.style.color = "#ccc";
                infoPanel.style.lineHeight = "1.3";
                infoPanel.style.flexShrink = "0";
                infoPanel.style.overflow = "hidden";
                infoPanel.innerHTML = '<span style="color: #888;">Gaussian splat info will appear here after execution</span>';

                // Add iframe and info panel to container
                container.appendChild(iframe);
                container.appendChild(infoPanel);

                // Add widget with required options
                const widget = this.addDOMWidget("preview_gaussian", "GAUSSIAN_PREVIEW", container, {
                    getValue() { return ""; },
                    setValue(v) { }
                });

                widget.computeSize = function () {
                    const isNodeResizeClamp = app.canvas?.resizing_node === node;
                    const width = Math.max(240, node.size[0] - 20);
                    if (isNodeResizeClamp) {
                        return [width, 80];
                    }
                    const top = this.last_y ?? 120;
                    // Adjusted offset from -8 to -20 to keep clear of the bottom resize handle
                    const height = Math.max(80, node.size[1] - top - 20);
                    return [width, height];
                };

                // Store references
                this.gaussianViewerIframe = iframe;
                this.gaussianInfoPanel = infoPanel;

				// Auto-focus iframe on hover so its internal shortcuts work immediately
                container.addEventListener("mouseenter", () => {
                    if (iframe.contentWindow) {
                        iframe.contentWindow.focus();
                    }
                });

                // Track iframe load state
                let iframeLoaded = false;
                iframe.addEventListener('load', () => {
                    iframeLoaded = true;
                });

                // Listen for messages from iframe
                window.addEventListener('message', async (event) => {
                    // Safety check: ignore messages not coming from this node's iframe
                    if (event.source !== iframe.contentWindow) {
                        return;
                    }

                    // Handle error messages from iframe
                    if (event.data.type === 'MESH_ERROR' && event.data.error) {
                        if (infoPanel) {
                            infoPanel.innerHTML = `<div style="color: #ff6b6b;">Error: ${event.data.error}</div>`;
                        }
                    }

                    // Handle camera lock and canvas snapshot upload
                    if (event.data.type === 'SET_CAMERA') {
                        const cameraWidget = node.widgets?.find(w => w.name === "camera_parameters");
                        const pathWidget = node.widgets?.find(w => w.name === "locked_image_path");

                        // If payload was cleared (user unlocked view)
                        if (event.data.camera === null) {
                            if (cameraWidget) cameraWidget.value = "{}";
                            if (pathWidget) pathWidget.value = "";
                            console.log("[DJBFilmz.GaussianPreview] Camera lock cleared");
                        } else {
                            if (cameraWidget) {
                                cameraWidget.value = JSON.stringify(event.data.camera);
                            }

                            if (event.data.image_data) {
                                try {
                                    // Convert base64 preview back to a raw blob
                                    const res = await fetch(event.data.image_data);
                                    const blob = await res.blob();

                                    const formData = new FormData();
                                    formData.append("image", blob, `gaussian_lock_${Date.now()}.png`);
                                    formData.append("type", "temp");
                                    formData.append("overwrite", "true");

                                    // Upload image directly to ComfyUI's web server
                                    const uploadResponse = await fetch("/upload/image", {
                                        method: "POST",
                                        body: formData
                                    });

                                    if (uploadResponse.ok) {
                                        const uploadResult = await uploadResponse.json();
                                        if (pathWidget) {
                                            pathWidget.value = uploadResult.name;
                                            console.log("[DJBFilmz.GaussianPreview] Lock image uploaded:", uploadResult.name);
                                        }
                                    } else {
                                        console.error("[DJBFilmz.GaussianPreview] Failed to upload locked image snapshot");
                                    }
                                } catch (err) {
                                    console.error("[DJBFilmz.GaussianPreview] Error handling snapshot upload:", err);
                                }
                            }
                        }
                        app.canvas.setDirty(true);
                    }
                });

                // Handle execution
                const onExecuted = this.onExecuted;
                this.onExecuted = function (message) {
                    onExecuted?.apply(this, arguments);

                    // Check for errors
                    if (message?.error && message.error[0]) {
                        infoPanel.innerHTML = `<div style="color: #ff6b6b;">Error: ${message.error[0]}</div>`;
                        return;
                    }

                    // The message IS the UI data (not message.ui)
                    if (message?.ply_path && message.ply_path[0]) {
                        const filename = message.filename?.[0];
                        const fileSizeMb = message.file_size_mb?.[0] || 'N/A';
                        const subfolder = message.subfolder?.[0] || "";
                        const type = message.type?.[0] || "output";
                        const previewFilename = message.preview_filename?.[0] || filename;
                        const previewSubfolder = message.preview_subfolder?.[0] || subfolder;
                        const previewType = message.preview_type?.[0] || type;
                        const previewSizeMb = message.preview_file_size_mb?.[0] || fileSizeMb;
                        const previewFormat = message.preview_format?.[0] || "ply";
                        const coordinateBasis = normalizeCoordinateBasis(message.coordinate_basis?.[0]);

                        // Extract camera parameters if provided
                        const extrinsics = message.extrinsics?.[0] || null;
                        const intrinsics = message.intrinsics?.[0] || null;

                        // Update info panel
                        infoPanel.innerHTML = `
                            <div style="display: grid; grid-template-columns: auto 1fr; gap: 2px 8px;">
                                <span style="color: #888;">File:</span>
                                <span style="color: #6cc;">${filename}</span>
                                <span style="color: #888;">Size:</span>
                                <span>${fileSizeMb} MB</span>
                                <span style="color: #888;">Preview:</span>
                                <span>${previewFormat.toUpperCase()} · ${previewSizeMb} MB</span>
                            </div>
                        `;

                        // ComfyUI serves output files via /view API endpoint
                        const filepath = `/view?filename=${encodeURIComponent(previewFilename)}&type=${encodeURIComponent(previewType)}&subfolder=${encodeURIComponent(previewSubfolder)}`;

                        // Function to fetch and send data to iframe
                        const fetchAndSend = async () => {
                            if (!iframe.contentWindow) {
                                return;
                            }

                            try {
                                // Fetch the preview file from parent context (authenticated)
                                const response = await fetch(filepath);
                                if (!response.ok) {
                                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                                }
                                const arrayBuffer = await response.arrayBuffer();

                                // Fetch active camera lock parameter widget value to preserve view
                                const cameraWidget = node.widgets?.find(w => w.name === "camera_parameters");
                                const cameraParamsVal = cameraWidget ? cameraWidget.value : "{}";

                                // Send the data to iframe with camera parameters and active locks
                                iframe.contentWindow.postMessage({
                                    type: "LOAD_MESH_DATA",
                                    data: arrayBuffer,
                                    filename: previewFilename,
                                    sourceFilename: filename,
                                    format: previewFormat,
                                    extrinsics: extrinsics,
                                    intrinsics: intrinsics,
                                    coordinateBasis: coordinateBasis,
                                    camera_parameters: cameraParamsVal,
                                    timestamp: Date.now()
                                }, "*", [arrayBuffer]);
                            } catch (error) {
                                console.error("[DJBFilmz.GaussianPreview] Error fetching preview data:", error);
                                infoPanel.innerHTML = `<div style="color: #ff6b6b;">Error loading preview: ${error.message}</div>`;
                            }
                        };

                        // Fetch and send when iframe is ready
                        if (iframeLoaded) {
                            fetchAndSend();
                        } else {
                            setTimeout(fetchAndSend, 500);
                        }
                    }
                };

                return r;
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (data) {
                const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
                ensureCoordinateBasisWidget(this, getConfiguredCoordinateBasis(this, data));
                return r;
            };

            const onSerialize = nodeType.prototype.onSerialize;
            nodeType.prototype.onSerialize = function (data) {
                const r = onSerialize ? onSerialize.apply(this, arguments) : undefined;
                ensureCoordinateBasisWidget(this);
                const widget = this.widgets?.find((w) => w.name === "coordinate_basis");
                if (widget && data?.widgets_values) {
                    const widgetIndex = this.widgets.indexOf(widget);
                    if (widgetIndex >= 0) {
                        data.widgets_values[widgetIndex] = widget.value || "auto";
                    }
                }
                data.properties = data.properties || {};
                data.properties.coordinate_basis = widget?.value || "auto";
                return r;
            };
        }
    }
});