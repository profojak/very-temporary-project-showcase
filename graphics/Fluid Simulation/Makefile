all: release

release: setup-release
	meson compile -C release
	mv release/fluid .

debug: setup-debug
	meson compile -C debug
	mv debug/fluid .

setup-release:
	meson setup release --buildtype=release

setup-debug:
	meson setup debug --buildtype=debugoptimized

run: release
	./fluid config.json

clean:
	@rm -rf release debug fluid
