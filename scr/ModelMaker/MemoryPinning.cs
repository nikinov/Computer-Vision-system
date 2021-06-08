using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace Wision.CppRoutinesWrapper
{
	public struct MemoryPinning : IDisposable
	{
		private readonly GCHandle[] _handles;

		public MemoryPinning(params object[] objs)
		{
			_handles = objs.Select(obj => GCHandle.Alloc(obj, GCHandleType.Pinned)).ToArray();
		}

		public void Unpin()
		{
			Dispose();
		}

		public IntPtr[] GetPinnedAddresses()
		{
			return _handles.Select(h => h.AddrOfPinnedObject()).ToArray();
		}

		#region IDisposable Members

		public void Dispose()
		{
			foreach (var handle in _handles)
				if (handle.Target != null)
				    handle.Free();
		}

		#endregion
	}

	public struct PinnedObject : IDisposable
	{
		private readonly GCHandle _handle;

		internal PinnedObject(object obj)
		{
			_handle = GCHandle.Alloc(obj, GCHandleType.Pinned);
		}

		public void Unpin()
		{
			Dispose();
		}

		#region IDisposable Members

		public void Dispose()
		{
			if (_handle.Target != null)
			    _handle.Free();
		}

		#endregion

		public static implicit operator IntPtr(PinnedObject pinned)
		{
			return pinned._handle.AddrOfPinnedObject();
		}
	}

	public struct PinnedObjectArray : IDisposable
	{
		private readonly GCHandle[] _handles;
		private readonly IntPtr[] _pointers;

		internal PinnedObjectArray(object[] objects)
		{
			_handles = objects.Select(obj => GCHandle.Alloc(obj, GCHandleType.Pinned)).ToArray();
			_pointers = _handles.Select(h => h.AddrOfPinnedObject()).ToArray();
		}

		public void Unpin()
		{
			Dispose();
		}

		#region IDisposable Members
		public void Dispose()
		{
			foreach (var handle in _handles)
				if (handle.Target != null)
                    handle.Free();
		}
		#endregion

		public static implicit operator IntPtr[](PinnedObjectArray pinned)
		{
			return pinned._pointers;
		}
	}

	public static class PinnedObjectExtensions
	{
		public static PinnedObject Pin(this object obj)
		{
			return new PinnedObject(obj);
		}

		public static PinnedObjectArray Pin(this object[] objects)
		{
			return new PinnedObjectArray(objects);
		}
	}
}
