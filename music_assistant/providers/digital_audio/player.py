
root@d5369777-music-assistant-dev:/# sed -n '220,280p' /app/venv/lib/python3.14/site-packages/music_assistant/providers/multichannel_audio/player.py
        """
        from .pa_simple import PASimpleStream  # noqa: PLC0415
        from music_assistant.helpers.ffmpeg import FFMpeg  # noqa: PLC0415

        if source_channels == 0:
            source_channels = self.channels

        output_format = AudioFormat(
            content_type=ContentType.from_bit_depth(self.bit_depth),
            sample_rate=self.sample_rate,
            bit_depth=self.bit_depth,
            channels=source_channels,
        )
        self.logger.debug(
            "Requesting output format: %dch %dHz %dbit %s (source=%d player=%d)",
            output_format.channels,
            output_format.sample_rate,
            output_format.bit_depth,
            output_format.content_type,
            source_channels,
            self.channels,
        )

        # Target 10ms chunks to ensure steady delivery to PA sinks.
        # Default get_ffmpeg_stream chunks are too large causing delivery gaps.
        chunk_size = int(self.sample_rate * 0.010) * source_channels * 4
        chunk_size = max((chunk_size // 4) * 4, 4 * source_channels * 4)

        streams: dict[str, PASimpleStream] = {}
        ffmpeg_proc: FFMpeg | None = None
        try:
            # Only open PA streams for pairs whose channel indices exist in the source
            for sink_name, (left_idx, right_idx) in self._pair_sinks.items():
                if left_idx >= source_channels or right_idx >= source_channels:
                    continue
                sname = sink_name
                stream = await self.mass.loop.run_in_executor(
                    None,
                    lambda s=sname: PASimpleStream(
                        sink_name=s,
                        app_name="music-assistant-multichannel",
                        rate=self.sample_rate,
                        channels=2,
                        bit_depth=self.bit_depth,
                        buffer_msec=80,
                    ),
                )
                streams[sink_name] = stream
                self.logger.debug("Opened PA stream for %s", sink_name)
            self.logger.info(
                "Multichannel playback started: %d active pairs, %dch source, %dHz, %dbit",
                len(streams),
                source_channels,
                self.sample_rate,
                self.bit_depth,
            )

            ffmpeg_proc = FFMpeg(
                audio_input=url,
                input_format=AudioFormat(content_type=ContentType.UNKNOWN),
                output_format=output_format,
root@d5369777-music-assistant-dev:/# 
